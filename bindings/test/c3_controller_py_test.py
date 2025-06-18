import numpy as np
from pydrake.geometry import Meshcat
from pyc3 import (
    C3Controller,
    LCSSimulator,
    LoadC3ControllerOptions,
)
from test_utils import C3Solution2Input, Vector2TimestampedVector
from bindings.test.c3_core_py_test import (
    make_cartpole_with_soft_walls_dynamics,
    make_cartpole_costs,
)

from pydrake.all import (
    DiagramBuilder,
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
    MultibodyPositionToGeometryPose,
    Demultiplexer,
)


def AddVisualizer(builder, scene_graph, state_port, time_step=0.0):
    assert builder is not None and scene_graph is not None
    plant = MultibodyPlant(time_step)
    parser = Parser(plant, scene_graph)

    # Load the Cartpole model from an SDF file.
    file = "systems/test/res/cartpole_softwalls.sdf"
    parser.AddModels(file)
    plant.Finalize()

    # Add a Demultiplexer to split the state vector.
    demux = builder.AddSystem(Demultiplexer(4, 2))
    builder.Connect(state_port, demux.get_input_port())

    # Add a MultibodyPositionToGeometryPose system to convert state to geometry pose.
    to_geometry_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))

    # Connect the output ports to the SceneGraph.
    builder.Connect(demux.get_output_port(0), to_geometry_pose.get_input_port())
    builder.Connect(
        to_geometry_pose.get_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()),
    )

    return plant


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
        ZeroOrderHold(1 / options.publish_frequency, cartpole.num_states())
    )

    # Connect simulator and ZeroOrderHold.
    builder.Connect(
        lcs_simulator.get_output_port_next_state(),
        state_zero_order_hold.get_input_port(),
    )
    builder.Connect(
        state_zero_order_hold.get_output_port(), lcs_simulator.get_input_port_state()
    )

    # Add cartpole geometry.
    plant = AddVisualizer(builder, scene_graph, state_zero_order_hold.get_output_port())

    # Add the C3 controller.
    c3_controller = builder.AddSystem(C3Controller(plant, costs, options))

    # Add constant value source for the LCS system.
    lcs = builder.AddSystem(ConstantValueSource(Value(cartpole)))

    # Add constant vector source for the desired state.
    n = cartpole.num_states()
    xdes = builder.AddSystem(ConstantVectorSource(np.zeros((n, 1))))

    # Add vector-to-timestamped-vector converter.
    vector_to_timestamped_vector = builder.AddSystem(Vector2TimestampedVector(n))
    builder.Connect(
        state_zero_order_hold.get_output_port(),
        vector_to_timestamped_vector.get_input_port(0),
    )

    # Connect controller inputs.
    builder.Connect(
        vector_to_timestamped_vector.get_output_port(0),
        c3_controller.get_input_port_lcs_state(),
    )
    builder.Connect(lcs.get_output_port(0), c3_controller.get_input_port_lcs())
    builder.Connect(xdes.get_output_port(), c3_controller.get_input_port_target())

    # Add and connect C3 solution action system.
    c3_action = builder.AddSystem(C3Solution2Input(cartpole.num_inputs()))
    builder.Connect(
        c3_controller.get_output_port_c3_solution(), c3_action.get_input_port(0)
    )
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
    state_context = diagram.GetMutableSubsystemContext(
        state_zero_order_hold, diagram_context
    )
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
