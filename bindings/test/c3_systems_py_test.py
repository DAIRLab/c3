import numpy as np
from pydrake.geometry import Meshcat
from pyc3 import C3Controller, LCSSimulator, TimestampedVector, C3Solution,LoadC3ControllerOptions
from c3_py_test import make_cartpole_with_soft_walls_dynamics, make_cartpole_costs

from pydrake.all import (
    AbstractValue,
    BasicVector,
    DiagramBuilder,
    LeafSystem,
    SceneGraph,
    Simulator,
    ZeroOrderHold,
    ConstantValueSource,
    ConstantVectorSource,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Value,
    MultibodyPlant,
    Parser,
    MultibodyPositionToGeometryPose
)

class CartpoleGeometry(LeafSystem):
    def __init__(self, scene_graph, time_step=0.0):
        super().__init__()
        self.plant = MultibodyPlant(time_step)
        parser = Parser(self.plant, scene_graph)

        # Load the Cartpole model from an SDF file.
        file = "systems/test/res/cartpole_softwalls.sdf"
        parser.AddModels(file)
        self.plant.Finalize()

        # Declare input and output ports.
        self.state_input_port = self.DeclareVectorInputPort("state", BasicVector(4)).get_index()
        self.scene_graph_state_output_port = self.DeclareVectorOutputPort(
            "scene_graph_state", BasicVector(2), self.OutputGeometryPose
        ).get_index()

    def get_input_port_state(self):
        return self.get_input_port(self.state_input_port)

    def get_output_port_scene_graph_state(self):
        return self.get_output_port(self.scene_graph_state_output_port)

    @staticmethod
    def AddToBuilder(builder, scene_graph, state_port, time_step=0.0):
        geometry = builder.AddSystem(CartpoleGeometry(scene_graph, time_step))

        # Connect the state port to the input port of the CartpoleGeometry system.
        builder.Connect(state_port, geometry.get_input_port_state())

        # Add a MultibodyPositionToGeometryPose system to convert state to geometry pose.
        to_geometry_pose = builder.AddSystem(MultibodyPositionToGeometryPose(geometry.plant))

        # Connect the output ports to the SceneGraph.
        builder.Connect(geometry.get_output_port_scene_graph_state(), to_geometry_pose.get_input_port())
        builder.Connect(
            to_geometry_pose.get_output_port(),
            scene_graph.get_source_pose_port(geometry.plant.get_source_id())
        )

        return geometry

    def OutputGeometryPose(self, context, output):
        state = self.EvalVectorInput(context, self.state_input_port).get_value()
        output.SetFromVector(state[:2])  # Map state to 2D vector for SceneGraph.

class C3Solution2Input(LeafSystem):
    def __init__(self):
        super().__init__()
        # Declare input port for C3 solutions.
        self.c3_solution_port_index = self.DeclareAbstractInputPort(
            "c3_solution", Value(C3Solution())
        ).get_index()
        # Declare output port for actions.
        self.c3_action_port_index = self.DeclareVectorOutputPort(
            "u", BasicVector(1), self.GetC3Action
        ).get_index()

    def GetC3Action(self, context, output):
        input_value = self.EvalAbstractInput(context, 0)
        assert input_value is not None
        sol = input_value.get_value()
        output[0] = sol.u_sol[0][0]  # Set action output.


class Vector2TimestampedVector(LeafSystem):
    def __init__(self):
        super().__init__()
        self.vector_port_index = self.DeclareVectorInputPort("state", BasicVector(4)).get_index()
        self.timestamped_vector_port_index = self.DeclareVectorOutputPort(
            "timestamped_state",
            TimestampedVector(4),
            self.Convert,
        ).get_index()

    def Convert(self, context, output):
        input_vector = self.EvalVectorInput(context, self.vector_port_index).get_value()
        output.SetDataVector(input_vector)
        output.set_timestamp(context.get_time())  # Set timestamp.


def DoMain():
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())

    # Initialize the C3 cartpole problem.
    options = LoadC3ControllerOptions(
        "/home/stephen/Workspace/DAIR/c3/core/test/res/c3_cartpole_options.yaml"
    )
    cartpole = make_cartpole_with_soft_walls_dynamics(options.N)
    costs = make_cartpole_costs(cartpole, options)

    # Add the LCS simulator.
    lcs_simulator = builder.AddSystem(LCSSimulator(cartpole))

    # Add a ZeroOrderHold system for state updates.
    state_zero_order_hold = builder.AddSystem(
        ZeroOrderHold(1 / options.publish_frequency, 4)
    )

    # Connect simulator and ZeroOrderHold.
    builder.Connect(lcs_simulator.get_output_port_next_state(), state_zero_order_hold.get_input_port())
    builder.Connect(state_zero_order_hold.get_output_port(), lcs_simulator.get_input_port_state())

    # Add cartpole geometry.
    geometry = CartpoleGeometry.AddToBuilder(
        builder, scene_graph, state_zero_order_hold.get_output_port(), 0.0
    )

    # Add the C3 controller.
    c3_controller = builder.AddSystem(C3Controller(geometry.plant, costs, options))

    # Add constant value source for the LCS system.
    lcs = builder.AddSystem(ConstantValueSource(Value(cartpole)))

    # Add constant vector source for the desired state.
    n = cartpole.num_states()
    xdes = builder.AddSystem(ConstantVectorSource(np.zeros((n, 1))))

    # Add vector-to-timestamped-vector converter.
    vector_to_timestamped_vector = builder.AddSystem(Vector2TimestampedVector())
    builder.Connect(state_zero_order_hold.get_output_port(), vector_to_timestamped_vector.get_input_port(0))

    # Connect controller inputs.
    builder.Connect(
        vector_to_timestamped_vector.get_output_port(0), c3_controller.get_input_port_lcs_state()
    )
    builder.Connect(lcs.get_output_port(0), c3_controller.get_input_port_lcs())
    builder.Connect(xdes.get_output_port(), c3_controller.get_input_port_target())

    # Add and connect C3 solution action system.
    c3_action = builder.AddSystem(C3Solution2Input())
    builder.Connect(c3_controller.get_output_port_c3_solution(), c3_action.get_input_port(0))
    builder.Connect(c3_action.get_output_port(0), lcs_simulator.get_input_port_action())
    builder.Connect(lcs.get_output_port(), lcs_simulator.get_input_port_lcs())

    # Set up Meshcat visualizer.
    meshcat = Meshcat()
    params = MeshcatVisualizerParams()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    # Build the system diagram.
    diagram = builder.Build()

    # Create a default context for the diagram.
    diagram_context = diagram.CreateDefaultContext()

    # Set initial positions for the state subsystem.
    state_context = diagram.GetMutableSubsystemContext(state_zero_order_hold, diagram_context)
    x0 = np.array([0, -0.5, 0.5, -0.4])
    state_zero_order_hold.SetVectorState(state_context, x0)

    # Create and configure the simulator.
    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(True)
    simulator.set_target_realtime_rate(1.0)  # Run simulation at real-time speed.
    simulator.Initialize()
    simulator.AdvanceTo(10.0)  # Run simulation for 10 seconds.

    return -1  # Indicate end of program.


if __name__ == "__main__":
    DoMain()
