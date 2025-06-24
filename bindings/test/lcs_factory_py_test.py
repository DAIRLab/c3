import numpy as np
from tqdm import tqdm
from pyc3 import (
    LCS,
    LCSFactory,
    C3MIQP,
    LoadC3ControllerOptions,
    ConstraintVariable,
)

import matplotlib.pyplot as plt

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
    System,
)


def make_cube_pivoting_lcs_plant():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant, scene_graph)
    parser.AddModels("systems/test/resources/cube_pivoting/cube_pivoting.sdf")
    plant.Finalize()

    # Build the plant diagram.
    plant_diagram = builder.Build()
    plant_diagram_context = plant_diagram.CreateDefaultContext()

    # Retrieve collision geometries for relevant bodies.
    platform_collision_geoms = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("platform")
    )
    cube_collision_geoms = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("cube")
    )
    left_finger_collision_geoms = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("left_finger")
    )
    right_finger_collision_geoms = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("right_finger")
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

    return builder, plant_diagram, plant_diagram_context, plant, contact_pairs


def main():
    c3_controller_options = LoadC3ControllerOptions("systems/test/resources/cube_pivoting/c3_controller_pivoting_options.yaml")
    c3_options = c3_controller_options.c3_options
    lcs_factory_options = c3_controller_options.lcs_factory_options
    _, diagram, diagram_context, plant, contact_pairs = make_cube_pivoting_lcs_plant()

    plant_autodiff = System.ToAutoDiffXd(plant)
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant_autodiff_context = plant_autodiff.CreateDefaultContext()
    pivoting_factory = LCSFactory(
        plant,
        plant_context,
        plant_autodiff,
        plant_autodiff_context,
        contact_pairs,
        lcs_factory_options,
    )

    costs = C3MIQP.CreateCostMatricesFromC3Options(c3_options, lcs_factory_options.N)

    x0 = np.array([0, 0.75, 0, -0.6, 0.75, 0.1, 0.125, 0, 0, 0, 0, 0, 0, 0])
    xd = np.array([0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0])
    xd = [xd for _ in range(lcs_factory_options.N + 1)]

    system_iter = 200

    x = np.zeros((x0.shape[0], system_iter + 1))
    u = np.zeros((plant.num_actuators(), system_iter + 1))

    x[:, 0] = x0.ravel()

    pivoting = LCS.CreatePlaceholderLCS(
        plant.num_positions() + plant.num_velocities(),
        plant.num_actuators(),
        LCSFactory.GetNumContactVariables(lcs_factory_options),
        lcs_factory_options.N,
        lcs_factory_options.dt,
    )
    opt = C3MIQP(pivoting, costs, xd, c3_options)

    # Add linear constraints to the controller.
    A = np.zeros((14, 14))
    A[3, 3] = 1
    A[4, 4] = 1
    A[5, 5] = 1
    A[6, 6] = 1
    lower_bound = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    upper_bound = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    opt.AddLinearConstraint(
        A, lower_bound, upper_bound, ConstraintVariable.STATE
    )  # Assuming ConstraintVariable is an enum

    for i in tqdm(range(system_iter)):
        pivoting_factory.UpdateStateAndInput(x[:, i], u[:, i])
        pivoting = pivoting_factory.GenerateLCS()
        opt.UpdateLCS(pivoting)
        opt.Solve(x[:, i])
        u[:, i + 1] = opt.GetInputSolution()[0]
        prediction = pivoting.Simulate(x[:, i], u[:, i + 1])
        x[:, i + 1] = prediction

    dt = lcs_factory_options.dt
    time_x = np.arange(0, system_iter * dt + dt, dt)

    plt.plot(time_x, x.T[:, : plant.num_positions()])
    plt.legend(plant.GetPositionNames())
    plt.show()


if __name__ == "__main__":
    main()
