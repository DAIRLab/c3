import argparse
import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    Simulator,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ContactVisualizer,
    ConstantVectorSource,
    ZeroOrderHold,
    SpatialForce,
    ExternallyAppliedSpatialForce,
    LeafSystem,
    System,
    AbstractValue,
)

from pyc3 import (
    C3,
    C3Controller,
    LCSFactorySystem,
    LoadC3ControllerOptions,
    ConstraintVariable,
)

from test_utils import C3Solution2Input, Vector2TimestampedVector
from bindings.test.c3_core_py_test import (
    make_cartpole_with_soft_walls_dynamics,
    make_cartpole_costs,
)


class SoftWallReactionForce(LeafSystem):
    def __init__(self, cartpole_softwalls):
        LeafSystem.__init__(self)
        self.cartpole_softwalls_ = cartpole_softwalls
        self.DeclareVectorInputPort("cartpole_state", 4)
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self.CalcSoftWallSpatialForce,
        )
        self.pole_body_ = cartpole_softwalls.GetBodyByName("Pole")
        self.wall_stiffness = 100

    def CalcSoftWallSpatialForce(self, context, output):
        cartpole_state = self.get_input_port(0).Eval(context)
        spatial_forces = []

        # Predefined values
        left_wall_xpos = -0.35
        right_wall_xpos = 0.35
        pole_length = 0.6

        # Calculate wall force
        pole_tip_xpos = cartpole_state[0] - pole_length * np.sin(cartpole_state[1])
        left_wall_force = max(0.0, left_wall_xpos - pole_tip_xpos) * self.wall_stiffness
        right_wall_force = (
            min(0.0, right_wall_xpos - pole_tip_xpos) * self.wall_stiffness
        )
        wall_force = 0.0
        if left_wall_force != 0:
            wall_force = left_wall_force
        elif right_wall_force != 0:
            wall_force = right_wall_force

        # Set force value
        spatial_force = ExternallyAppliedSpatialForce()
        spatial_force.body_index = self.pole_body_.index()
        spatial_force.p_BoBq_B = self.pole_body_.default_com()
        spatial_force.F_Bq_W = SpatialForce(
            np.array([0.0, 0.0, 0.0]), np.array([wall_force, 0.0, 0.0])  # no torque
        )
        spatial_forces.append(spatial_force)
        output.set_value(spatial_forces)


def RunCartpoleTest():
    # Initialize the C3 cartpole problem.
    options = LoadC3ControllerOptions(
        "/home/stephen/Workspace/DAIR/c3/core/test/res/c3_cartpole_options.yaml"
    )
    cartpole = make_cartpole_with_soft_walls_dynamics(options.N)
    costs = make_cartpole_costs(cartpole, options)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant, scene_graph)
    file = "systems/test/res/cartpole_softwalls.sdf"
    parser.AddModels(file)
    plant.Finalize()

    # LCS Factory System
    plant_diagram = builder.Build()
    plant_context = plant_diagram.CreateDefaultContext()
    plant_for_lcs = plant
    plant_for_lcs_context = plant_diagram.GetMutableSubsystemContext(
        plant, plant_context
    )
    plant_autodiff = System.ToAutoDiffXd(plant_for_lcs)
    plant_context_autodiff = plant_autodiff.CreateDefaultContext()

    left_wall_contact_points = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("left_wall")
    )

    right_wall_contact_points = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("right_wall")
    )

    pole_point_geoms = plant.GetCollisionGeometriesForBody(plant.GetBodyByName("Pole"))
    contact_geoms = {}
    contact_geoms["LEFT_WALL"] = left_wall_contact_points
    contact_geoms["RIGHT_WALL"] = right_wall_contact_points
    contact_geoms["POLE_POINT"] = pole_point_geoms

    contact_pairs = []
    contact_pairs.append(
        tuple(sorted((contact_geoms["LEFT_WALL"][0], contact_geoms["POLE_POINT"][0])))
    )
    contact_pairs.append(
        tuple(sorted((contact_geoms["RIGHT_WALL"][0], contact_geoms["POLE_POINT"][0])))
    )

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
    parser = Parser(plant, scene_graph)
    file = "systems/test/res/cartpole_softwalls.sdf"
    parser.AddModels(file)
    plant.set_penetration_allowance(0.15)
    plant.Finalize()

    lcs_factory_system = builder.AddSystem(
        LCSFactorySystem(
            plant_for_lcs,
            plant_for_lcs_context,
            plant_autodiff,
            plant_context_autodiff,
            contact_pairs,
            options,
        )
    )

    # Add the C3 controller.
    c3_controller = builder.AddSystem(C3Controller(plant_for_lcs, costs, options))

    # Add constant vector source for the desired state.
    n = cartpole.num_states()
    xdes = builder.AddSystem(ConstantVectorSource(np.zeros((n, 1))))

    # Add vector-to-timestamped-vector converter.
    vector_to_timestamped_vector = builder.AddSystem(Vector2TimestampedVector(4))
    builder.Connect(
        plant.get_state_output_port(),
        vector_to_timestamped_vector.get_input_port_state(),
    )

    # Connect controller inputs.
    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        c3_controller.get_input_port_lcs_state(),
    )
    builder.Connect(
        lcs_factory_system.get_output_port_lcs(), c3_controller.get_input_port_lcs()
    )
    builder.Connect(xdes.get_output_port(), c3_controller.get_input_port_target())

    # Add and connect C3 solution input system.
    c3_input = builder.AddSystem(C3Solution2Input(1))
    builder.Connect(
        c3_controller.get_output_port_c3_solution(),
        c3_input.get_input_port_c3_solution(),
    )
    builder.Connect(
        c3_input.get_output_port_c3_input(), plant.get_actuation_input_port()
    )

    # Add a ZeroOrderHold system for state updates.
    k = cartpole.num_inputs()
    input_zero_order_hold = builder.AddSystem(
        ZeroOrderHold(
            1 / options.publish_frequency,
            k,
        )
    )
    builder.Connect(
        c3_input.get_output_port_c3_input(), input_zero_order_hold.get_input_port()
    )

    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        lcs_factory_system.get_input_port_lcs_state(),
    )
    builder.Connect(
        input_zero_order_hold.get_output_port(),
        lcs_factory_system.get_input_port_lcs_input(),
    )

    # Add the SoftWallReactionForce system.
    soft_wall_reaction_force = builder.AddSystem(SoftWallReactionForce(plant_for_lcs))
    # Connect the SoftWallReactionForce system to the LCSFactorySystem.
    builder.Connect(
        plant.get_state_output_port(), soft_wall_reaction_force.get_input_port()
    )
    # Connect the SoftWallReactionForce output to the plant's external forces.
    builder.Connect(
        soft_wall_reaction_force.get_output_port(),
        plant.get_applied_spatial_force_input_port(),
    )

    # Set up Meshcat visualizer.
    meshcat = Meshcat()
    params = MeshcatVisualizerParams()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    diagram = builder.Build()

    # Create a default context for the diagram.
    diagram_context = diagram.CreateDefaultContext()

    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    x0 = np.array([0, -0.5, 0.5, -0.4])
    plant.SetPositionsAndVelocities(plant_context, x0)
    # Create and configure the simulator.
    simulator = Simulator(diagram, diagram_context)

    simulator.set_target_realtime_rate(0.25)  # Run simulation at real-time speed.
    simulator.Initialize()
    simulator.AdvanceTo(10.0)  # Run
    #   simulation for 10 seconds.

    return 1


def RunPivotingTest():
    # Build the plant and scene graph for the pivoting system.
    builder = DiagramBuilder()
    plant_for_lcs, scene_graph_for_lcs = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser_for_lcs = Parser(plant_for_lcs, scene_graph_for_lcs)
    file_for_lcs = "systems/test/res/cube_pivoting.sdf"
    parser_for_lcs.AddModels(file_for_lcs)
    plant_for_lcs.Finalize()

    # Build the plant diagram.
    plant_diagram = builder.Build()

    # Retrieve collision geometries for relevant bodies.
    platform_collision_geoms = plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("platform")
    )
    cube_collision_geoms = plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("cube")
    )
    left_finger_collision_geoms = plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("left_finger")
    )
    right_finger_collision_geoms = plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("right_finger")
    )

    # Map collision geometries to their respective components.
    contact_geoms = {}
    contact_geoms["PLATFORM"] = platform_collision_geoms
    contact_geoms["CUBE"] = cube_collision_geoms
    contact_geoms["LEFT_FINGER"] = left_finger_collision_geoms
    contact_geoms["RIGHT_FINGER"] = right_finger_collision_geoms

    # Define contact pairs for the LCS system.
    contact_pairs = []
    contact_pairs.append(
        tuple([contact_geoms["CUBE"][0], contact_geoms["LEFT_FINGER"][0]])
    )
    contact_pairs.append(
        tuple([contact_geoms["CUBE"][0], contact_geoms["PLATFORM"][0]])
    )
    contact_pairs.append(
        tuple([contact_geoms["CUBE"][0], contact_geoms["RIGHT_FINGER"][0]])
    )

    # Build the main diagram.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.01)
    parser = Parser(plant, scene_graph)
    file = "systems/test/res/cube_pivoting.sdf"
    parser.AddModels(file)
    plant.Finalize()

    # Load controller options and cost matrices.
    options = LoadC3ControllerOptions("systems/test/res/c3_pivoting_options.yaml")
    cost = C3.CreateCostMatricesFromC3Options(options, options.N)

    # Create contexts for the plant and LCS factory system.
    plant_diagram_context = plant_diagram.CreateDefaultContext()
    plant_autodiff = System.ToAutoDiffXd(plant_for_lcs)
    plant_for_lcs_context = plant_diagram.GetMutableSubsystemContext(
        plant_for_lcs, plant_diagram_context
    )
    plant_context_autodiff = plant_autodiff.CreateDefaultContext()

    # Add the LCS factory system.
    lcs_factory_system = builder.AddSystem(
        LCSFactorySystem(
            plant_for_lcs,
            plant_for_lcs_context,
            plant_autodiff,
            plant_context_autodiff,
            contact_pairs,
            options,
        )
    )

    # Add the C3 controller.
    c3_controller = builder.AddSystem(C3Controller(plant_for_lcs, cost, options))
    c3_controller.set_name("c3_controller")

    # Add linear constraints to the controller.
    A = np.zeros((14, 14))
    A[3, 3] = 1
    A[4, 4] = 1
    A[5, 5] = 1
    A[6, 6] = 1
    lower_bound = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    upper_bound = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    c3_controller.AddLinearConstraint(
        A, lower_bound, upper_bound, ConstraintVariable.STATE
    )  # Assuming ConstraintVariable is an enum

    # Add a constant vector source for the desired state.
    xd = np.array([0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0])
    xdes = builder.AddSystem(ConstantVectorSource(xd))

    # Add a vector-to-timestamped-vector converter.
    vector_to_timestamped_vector = builder.AddSystem(Vector2TimestampedVector(14))
    builder.Connect(
        plant.get_state_output_port(),
        vector_to_timestamped_vector.get_input_port_state(),
    )

    # Connect controller inputs.
    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        c3_controller.get_input_port_lcs_state(),
    )
    builder.Connect(
        lcs_factory_system.get_output_port_lcs(), c3_controller.get_input_port_lcs()
    )
    builder.Connect(xdes.get_output_port(), c3_controller.get_input_port_target())

    # Add and connect the C3 solution input system.
    c3_input = builder.AddSystem(C3Solution2Input(4))
    builder.Connect(
        c3_controller.get_output_port_c3_solution(),
        c3_input.get_input_port_c3_solution(),
    )
    builder.Connect(
        c3_input.get_output_port_c3_input(), plant.get_actuation_input_port()
    )

    # Add a ZeroOrderHold system for state updates.
    input_zero_order_hold = builder.AddSystem(
        ZeroOrderHold(1 / options.publish_frequency, 4)
    )
    builder.Connect(
        c3_input.get_output_port_c3_input(), input_zero_order_hold.get_input_port()
    )
    builder.Connect(
        vector_to_timestamped_vector.get_output_port_timestamped_state(),
        lcs_factory_system.get_input_port_lcs_state(),
    )
    builder.Connect(
        input_zero_order_hold.get_output_port(),
        lcs_factory_system.get_input_port_lcs_input(),
    )

    # Set up Meshcat visualizer.
    meshcat = Meshcat()
    params = MeshcatVisualizerParams()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    ContactVisualizer.AddToBuilder(builder, plant, meshcat)

    # Build the diagram.
    diagram = builder.Build()

    # Create a default context for the diagram.
    diagram_context = diagram.CreateDefaultContext()

    # Set the initial state of the system.
    x0 = np.array([0, 0.75, 0, -0.6, 0.75, 0.1, 0.125, 0, 0, 0, 0, 0, 0, 0])
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.SetPositionsAndVelocities(plant_context, x0)

    # Create and configure the simulator.
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1)  # Run simulation at real-time speed.
    simulator.Initialize()
    simulator.AdvanceTo(10.0)  # Run simulation for 10 seconds.

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LCSFactorySystem with different experiments."
    )
    parser.add_argument(
        "--experiment_type",
        default="cube_pivoting",
        choices=["cartpole_softwalls", "cube_pivoting"],
        help="The type of experiment to run.",
    )
    args = parser.parse_args()

    if args.experiment_type == "cartpole_softwalls":
        print("Running Cartpole Softwalls Test...")
        exit(RunCartpoleTest())
    elif args.experiment_type == "cube_pivoting":
        print("Running Cube Pivoting Test...")
        exit(RunPivotingTest())
    else:
        print(
            f"Unknown experiment type: {args.experiment_type}. Supported types are 'cartpole_softwalls' and 'cube_pivoting'."
        )
        exit(-1)
