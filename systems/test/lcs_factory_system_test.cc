// Includes for core controllers, simulators, and test problems.
#include "systems/lcs_factory_system.h"

#include <algorithm>
#include <cmath>

#include <drake/geometry/drake_visualizer.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/systems/primitives/multiplexer.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <gflags/gflags.h>

#include "core/test/c3_cartpole_problem.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/common/system_utils.hpp"
#include "systems/lcs_simulator.h"
#include "systems/test/test_utils.hpp"

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

DEFINE_string(experiment_type, "cube_pivoting",
              "The type of experiment to test the LCSFactorySystem with. "
              "Options: 'cartpole_softwalls [Frictionless Spring System]', "
              "'cube_pivoting [Stewart and Trinkle System]'");
DEFINE_string(diagram_path, "",
              "Path to store the diagram (.ps) for the system. If empty, will "
              "be ignored");

using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;
using c3::systems::LCSFactorySystem;
using c3::systems::LCSSimulator;

using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::DiagramBuilder;
using drake::systems::rendering::MultibodyPositionToGeometryPose;

class SoftWallReactionForce final : public drake::systems::LeafSystem<double> {
  // Converted to C++ from cartpole_softwall.py by Hien
  // This class provides the spatial reaction forces given the state of the
  // cartpole
 public:
  explicit SoftWallReactionForce(
      const MultibodyPlant<double>* cartpole_softwalls)
      : cartpole_softwalls_(cartpole_softwalls) {
    this->DeclareVectorInputPort("cartpole_state", 4);
    this->DeclareAbstractOutputPort(
        "spatial_forces", &SoftWallReactionForce::CalcSoftWallSpatialForce);
    pole_body_ = &cartpole_softwalls_->GetBodyByName("Pole");
  }

  double wall_stiffness = 100;

 private:
  void CalcSoftWallSpatialForce(
      const drake::systems::Context<double>& context,
      std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>*
          output) const {
    output->resize(1);
    (*output)[0].body_index = pole_body_->index();
    (*output)[0].p_BoBq_B = pole_body_->default_com();

    // Get Input
    const auto& cartpole_state = get_input_port(0).Eval(context);

    // Predefined values
    double left_wall_xpos = -0.35;
    double right_wall_xpos = 0.35;
    double pole_length = 0.6;

    // Calculate wall force
    double pole_tip_xpos =
        cartpole_state[0] - pole_length * sin(cartpole_state[1]);
    double left_wall_force =
        std::max(0.0, left_wall_xpos - pole_tip_xpos) * wall_stiffness;
    double right_wall_force =
        std::min(0.0, right_wall_xpos - pole_tip_xpos) * wall_stiffness;
    double wall_force = 0.0;
    if (left_wall_force != 0)
      wall_force = left_wall_force;
    else if (right_wall_force != 0)
      wall_force = right_wall_force;

    // Set force value
    (*output)[0].F_Bq_W = drake::multibody::SpatialForce<double>(
        drake::Vector3<double>::Zero() /* no torque */,
        drake::Vector3<double>(wall_force, 0, 0));
  }
  const MultibodyPlant<double>* cartpole_softwalls_{nullptr};
  const drake::multibody::RigidBody<double>* pole_body_;
};

int RunCartpoleTest() {
  // Initialize the C3 cartpole problem.
  auto c3_cartpole_problem = C3CartpoleProblem();

  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);
  const std::string file_for_lcs =
      "systems/test/resources/cartpole_softwalls/cartpole_softwalls.sdf";
  parser_for_lcs.AddModels(file_for_lcs);
  plant_for_lcs.Finalize();

  //   // LCS Factory System
  auto plant_diagram = plant_builder.Build();

  std::vector<drake::geometry::GeometryId> left_wall_contact_points =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("left_wall"));

  std::vector<drake::geometry::GeometryId> right_wall_contact_points =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("right_wall"));

  std::vector<drake::geometry::GeometryId> pole_point_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("Pole"));
  std::unordered_map<std::string, std::vector<drake::geometry::GeometryId>>
      contact_geoms;
  contact_geoms["LEFT_WALL"] = left_wall_contact_points;
  contact_geoms["RIGHT_WALL"] = right_wall_contact_points;
  contact_geoms["POLE_POINT"] = pole_point_geoms;

  std::vector<SortedPair<GeometryId>> contact_pairs;
  contact_pairs.emplace_back(contact_geoms["LEFT_WALL"][0],
                             contact_geoms["POLE_POINT"][0]);
  contact_pairs.emplace_back(contact_geoms["RIGHT_WALL"][0],
                             contact_geoms["POLE_POINT"][0]);

  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.01);
  Parser parser(&plant, &scene_graph);
  const std::string file =
      "systems/test/resources/cartpole_softwalls/cartpole_softwalls.sdf";
  parser.AddModels(file);
  plant.set_penetration_allowance(0.15);
  plant.Finalize();

  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "systems/test/resources/cartpole_softwalls/"
      "c3_controller_cartpole_options.yaml");

  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();

  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);

  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());

  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();
  auto lcs_factory_system = builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Add the C3 controller.
  auto c3_controller = builder.AddSystem<C3Controller>(
      plant_for_lcs, c3_cartpole_problem.cost, options);

  // Add constant vector source for the desired state.
  auto xdes = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      c3_cartpole_problem.xdesired.at(0));

  // Add vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(4);
  builder.Connect(plant.get_state_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());
  builder.Connect(lcs_factory_system->get_output_port_lcs(),
                  c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(1);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());
  builder.Connect(c3_input->get_output_port_c3_input(),
                  plant.get_actuation_input_port());

  // Add a ZeroOrderHold system for state updates.
  auto state_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, c3_cartpole_problem.k);
  builder.Connect(c3_input->get_output_port_c3_input(),
                  state_zero_order_hold->get_input_port());

  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  builder.Connect(state_zero_order_hold->get_output_port(),
                  lcs_factory_system->get_input_port_lcs_input());

  // Add the SoftWallReactionForce system.
  auto soft_wall_reaction_force =
      builder.AddSystem<SoftWallReactionForce>(&plant_for_lcs);
  // Connect the SoftWallReactionForce system to the LCSFactorySystem.
  builder.Connect(plant.get_state_output_port(),
                  soft_wall_reaction_force->get_input_port());
  // Connect the SoftWallReactionForce output to the plant's external forces.
  builder.Connect(soft_wall_reaction_force->get_output_port(),
                  plant.get_applied_spatial_force_input_port());

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));

  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat,
      drake::multibody::meshcat::ContactVisualizerParams());

  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  plant.SetPositionsAndVelocities(&plant_context, c3_cartpole_problem.x0);
  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));

  simulator.set_target_realtime_rate(
      0.25);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(10.0);  // Run
  //   simulation for 10 seconds.

  return 1;
}

int RunPivotingTest() {
  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);
  const std::string file_for_lcs =
      "systems/test/resources/cube_pivoting/cube_pivoting.sdf";
  parser_for_lcs.AddModels(file_for_lcs);
  plant_for_lcs.Finalize();

  // Build the plant diagram.
  auto plant_diagram = plant_builder.Build();

  // Retrieve collision geometries for relevant bodies.
  std::vector<drake::geometry::GeometryId> platform_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("platform"));
  std::vector<drake::geometry::GeometryId> cube_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("cube"));
  std::vector<drake::geometry::GeometryId> left_finger_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("left_finger"));
  std::vector<drake::geometry::GeometryId> right_finger_collision_geoms =
      plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("right_finger"));

  // Map collision geometries to their respective components.
  std::unordered_map<std::string, std::vector<drake::geometry::GeometryId>>
      contact_geoms;
  contact_geoms["PLATFORM"] = platform_collision_geoms;
  contact_geoms["CUBE"] = cube_collision_geoms;
  contact_geoms["LEFT_FINGER"] = left_finger_collision_geoms;
  contact_geoms["RIGHT_FINGER"] = right_finger_collision_geoms;

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs;
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["LEFT_FINGER"][0]);
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["PLATFORM"][0]);
  contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                             contact_geoms["RIGHT_FINGER"][0]);

  // Build the main diagram.
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, 0.01);
  Parser parser(&plant, &scene_graph);
  const std::string file =
      "systems/test/resources/cube_pivoting/cube_pivoting.sdf";
  parser.AddModels(file);
  plant.Finalize();

  // Load controller options and cost matrices.
  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "systems/test/resources/cube_pivoting/"
      "c3_controller_pivoting_options.yaml");
  C3::CostMatrices cost = C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Add the LCS factory system.
  auto lcs_factory_system = builder.AddSystem<LCSFactorySystem>(
      plant_for_lcs, plant_for_lcs_context, *plant_autodiff,
      *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Add the C3 controller.
  auto c3_controller =
      builder.AddSystem<C3Controller>(plant_for_lcs, cost, options);
  c3_controller->set_name("c3_controller");

  // Add linear constraints to the controller.
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(14, 14);
  A(3, 3) = 1;
  A(4, 4) = 1;
  A(5, 5) = 1;
  A(6, 6) = 1;
  Eigen::VectorXd lower_bound(14);
  Eigen::VectorXd upper_bound(14);
  lower_bound << 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  upper_bound << 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0;
  c3_controller->AddLinearConstraint(A, lower_bound, upper_bound,
                                     ConstraintVariable::STATE);

  // Add a constant vector source for the desired state.
  Eigen::VectorXd xd(14);
  xd << 0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0;
  auto xdes =
      builder.AddSystem<drake::systems::ConstantVectorSource<double>>(xd);

  // Add a vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(14);
  builder.Connect(plant.get_state_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());
  builder.Connect(lcs_factory_system->get_output_port_lcs(),
                  c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect the C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(4);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());
  builder.Connect(c3_input->get_output_port_c3_input(),
                  plant.get_actuation_input_port());

  // Add a ZeroOrderHold system for state updates.
  auto input_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, 4);
  builder.Connect(c3_input->get_output_port_c3_input(),
                  input_zero_order_hold->get_input_port());
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  builder.Connect(input_zero_order_hold->get_output_port(),
                  lcs_factory_system->get_input_port_lcs_input());

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(params));
  drake::multibody::meshcat::ContactVisualizer<double>::AddToBuilder(
      &builder, plant, meshcat,
      drake::multibody::meshcat::ContactVisualizerParams());

  // Build the diagram.
  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  // Set the initial state of the system.
  Eigen::VectorXd x0(14);
  x0 << 0, 0.75, 0, -0.6, 0.75, 0.1, 0.125, 0, 0, 0, 0, 0, 0, 0;
  auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetPositionsAndVelocities(&plant_context, x0);

  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(1);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(10.0);  // Run simulation for 10 seconds.

  return 1;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_experiment_type == "cartpole_softwalls") {
    std::cout << "Running Cartpole Softwalls Test..." << std::endl;
    return RunCartpoleTest();
  } else if (FLAGS_experiment_type == "cube_pivoting") {
    std::cout << "Running Cube Pivoting Test..." << std::endl;
    return RunPivotingTest();
  } else {
    std::cerr
        << "Unknown experiment type: " << FLAGS_experiment_type
        << ". Supported types are 'cartpole_softwalls' and 'cube_pivoting'."
        << std::endl;
    return -1;
  }
}