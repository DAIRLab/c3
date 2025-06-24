// Includes for core controllers, simulators, and test problems.
#include "systems/c3_controller.h"

#include "core/test/c3_cartpole_problem.hpp"
#include "systems/c3_controller_options.h"
#include "systems/common/system_utils.hpp"
#include "systems/lcs_simulator.h"
#include "systems/test/test_utils.hpp"

// Includes for Drake systems and primitives.
#include <drake/geometry/drake_visualizer.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/systems/primitives/pass_through.h>
#include <drake/systems/primitives/vector_log_sink.h>
#include <drake/systems/primitives/zero_order_hold.h>

#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;
using c3::systems::LCSSimulator;
using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::DiagramBuilder;
using drake::systems::rendering::MultibodyPositionToGeometryPose;
using Eigen::VectorXd;

// Adds the Visualizer for the cartpole with softwalls system to a
// DiagramBuilder and connects it to the SceneGraph.
std::unique_ptr<MultibodyPlant<double>> AddVisualizer(
    drake::systems::DiagramBuilder<double>* builder,
    SceneGraph<double>* scene_graph,
    const drake::systems::OutputPort<double>& state_port,
    double time_step = 0.0) {
  DRAKE_DEMAND(builder != nullptr && scene_graph != nullptr);
  auto plant = std::make_unique<MultibodyPlant<double>>(0.0);
  Parser parser(plant.get(), scene_graph);

  // Load the Cartpole model from an SDF file.
  const std::string file = "systems/test/resources/cartpole_softwalls/cartpole_softwalls.sdf";
  parser.AddModels(file);
  plant->Finalize();

  // Connect the state port to the demultiplexer and then to the input port of
  // the CartpoleWithSoftWallsVisualizer system.
  auto demux = builder->AddSystem<drake::systems::Demultiplexer<double>>(4, 2);
  builder->Connect(state_port, demux->get_input_port());

  // Add a MultibodyPositionToGeometryPose system to convert state to
  // geometry pose.
  auto to_geometry_pose =
      builder->AddSystem<MultibodyPositionToGeometryPose<double>>(*plant);

  // Connect the output ports to the SceneGraph.
  builder->Connect(demux->get_output_port(0),
                   to_geometry_pose->get_input_port());
  builder->Connect(
      to_geometry_pose->get_output_port(),
      scene_graph->get_source_pose_port(plant->get_source_id().value()));

  return plant;
}

// Main function to build and run the simulation.
int DoMain() {
  DiagramBuilder<double> builder;
  SceneGraph<double>* scene_graph = builder.AddSystem<SceneGraph>();

  // Initialize the C3 cartpole problem.
  auto c3_cartpole_problem = C3CartpoleProblem();

  // Add the LCS simulator.
  auto lcs_simulator =
      builder.AddSystem<LCSSimulator>(*(c3_cartpole_problem.pSystem));

  C3ControllerOptions options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
      "systems/test/resources/cartpole_softwalls/c3_controller_cartpole_options.yaml");

  // Add a ZeroOrderHold system for state updates.
  auto state_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, c3_cartpole_problem.n);

  // Connect simulator and ZeroOrderHold.
  builder.Connect(lcs_simulator->get_output_port_next_state(),
                  state_zero_order_hold->get_input_port());
  builder.Connect(state_zero_order_hold->get_output_port(),
                  lcs_simulator->get_input_port_state());

  // Add cartpole geometry.
  auto plant = AddVisualizer(&builder, scene_graph,
                             state_zero_order_hold->get_output_port());

  // Add the C3 controller.
  auto c3_controller = builder.AddSystem<C3Controller>(
      *plant, c3_cartpole_problem.cost, options);

  // Add constant value source for the LCS system.
  auto lcs = builder.AddSystem<drake::systems::ConstantValueSource>(
      drake::Value<c3::LCS>(*(c3_cartpole_problem.pSystem)));

  // Add constant vector source for the desired state.
  auto xdes = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      c3_cartpole_problem.xdesired.at(0));

  // Add vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>(4);
  builder.Connect(state_zero_order_hold->get_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect controller inputs.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());
  builder.Connect(lcs->get_output_port(), c3_controller->get_input_port_lcs());
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add and connect C3 solution input system.
  auto c3_input = builder.AddSystem<C3Solution2Input>(1);
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_input->get_input_port_c3_solution());
  builder.Connect(c3_input->get_output_port_c3_input(),
                  lcs_simulator->get_input_port_action());
  builder.Connect(lcs->get_output_port(), lcs_simulator->get_input_port_lcs());

  // Set up Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, *scene_graph, meshcat, std::move(params));

  // Build the system diagram.
  auto diagram_ = builder.Build();

  // Create a default context for the diagram.
  auto diagram_context = diagram_->CreateDefaultContext();

  // Set initial positions for the state subsystem.
  auto& state_context = diagram_->GetMutableSubsystemContext(
      *state_zero_order_hold, diagram_context.get());
  state_zero_order_hold->SetVectorState(&state_context, c3_cartpole_problem.x0);

  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram_,
                                              std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(
      1.0);  // Run simulation at real-time speed.
  simulator.Initialize();
  simulator.AdvanceTo(20.0);  // Run simulation for 10 seconds.

  return -1;  // Indicate end of program.
}

// Main entry point of the program.
int main(int argc, char** argv) { return DoMain(); }
