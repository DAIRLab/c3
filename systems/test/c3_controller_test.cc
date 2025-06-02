// Includes for core controllers, simulators, and test problems.
#include "systems/c3_controller.h"

#include "systems/lcs_simulator.h"
#include "core/test/c3_cartpole_problem.hpp"
#include "systems/test/cartpole_geometry.hpp"

// Includes for Drake systems and primitives.
#include <drake/systems/primitives/pass_through.h>
#include <drake/systems/primitives/vector_log_sink.h>
#include <drake/systems/primitives/zero_order_hold.h>

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"

// Standard library includes.
#include <fstream>
#include <regex>
#include <string>

using c3::systems::C3Controller;
using c3::systems::LCSSimulator;
using c3::test::CartpoleGeometry;
using drake::geometry::SceneGraph;
using drake::systems::DiagramBuilder;
using Eigen::VectorXd;

// Custom LeafSystem to process C3 solutions and output actions.
class C3Solution2Input : public drake::systems::LeafSystem<double> {
 public:
  explicit C3Solution2Input() {
    // Declare input port for C3 solutions.
    c3_solution_port_index_ =
        this->DeclareAbstractInputPort("c3_solution",
                                       drake::Value<C3Output::C3Solution>())
            .get_index();
    // Declare output port for actions.
    c3_action_port_index_ =
        this->DeclareVectorOutputPort("u", 1, &C3Solution2Input::GetC3Action)
            .get_index();
  }

  // Getter for the input port.
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_index_);
  }

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_output_port_c3_action() const {
    return this->get_output_port(c3_action_port_index_);
  }

 private:
  drake::systems::InputPortIndex c3_solution_port_index_;
  drake::systems::OutputPortIndex c3_action_port_index_;

  // Method to compute the action from the C3 solution.
  void GetC3Action(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output) const {
    // Get the input C3 solution.
    const drake::AbstractValue* input = this->EvalAbstractInput(context, 0);
    DRAKE_ASSERT(input != nullptr);
    const auto& sol = input->get_value<C3Output::C3Solution>();
    // Set the action output based on the solution.
    output->SetAtIndex(0, sol.u_sol_(0));
  }
};

class Vector2TimestampedVector : public drake::systems::LeafSystem<double> {
 public:
  explicit Vector2TimestampedVector() {
    vector_port_index_ = this->DeclareVectorInputPort("state", 4).get_index();
    timestamped_vector_port_index_ =
        this->DeclareVectorOutputPort("timestamped_state",
                                      c3::systems::TimestampedVector<double>(4),
                                      &Vector2TimestampedVector::Convert)
            .get_index();
  }
  // Getter for the input port.
  const drake::systems::InputPort<double>& get_input_port_state() const {
    return this->get_input_port(vector_port_index_);
  }

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_output_port_timestamped_state()
      const {
    return this->get_output_port(timestamped_vector_port_index_);
  }

 private:
  drake::systems::InputPortIndex vector_port_index_;
  drake::systems::OutputPortIndex timestamped_vector_port_index_;

  void Convert(const drake::systems::Context<double>& context,
               c3::systems::TimestampedVector<double>* output) const {
    output->SetDataVector(
        this->EvalVectorInput(context, vector_port_index_)->get_value());
    // Set the timestamp to the current time.
    output->set_timestamp(context.get_time());
  }
};

// Utility function to save and visualize the system diagram.
void DrawAndSaveDiagramGraph(const drake::systems::Diagram<double>& diagram,
                             std::string path) {
  // Default path if none is provided.
  if (path.empty()) path = "../" + diagram.get_name();

  // Save Graphviz string to a file.
  std::ofstream out(path);
  out << diagram.GetGraphvizString();
  out.close();

  // Convert Graphviz string to an image file using the `dot` command.
  std::regex r(" ");
  path = std::regex_replace(path, r, "\\ ");
  std::string cmd = "dot -Tps " + path + " -o " + path + ".ps";
  (void)std::system(cmd.c_str());

  // Remove the temporary Graphviz string file.
  cmd = "rm " + path;
  (void)std::system(cmd.c_str());
}

// Main function to build and run the simulation.
int DoMain() {
  // Create a diagram builder and add a SceneGraph.
  DiagramBuilder<double> builder;
  SceneGraph<double>* scene_graph = builder.AddSystem<SceneGraph>();

  // Initialize the C3 cartpole problem.
  C3CartpoleProblem c3_cartpole_problem = C3CartpoleProblem();

  // Add the LCS simulator to the builder.
  LCSSimulator* lcs_simulator =
      builder.AddSystem<LCSSimulator>(c3_cartpole_problem.pSystem.get());

  // Add a ZeroOrderHold system for state updates.
  drake::systems::ZeroOrderHold<double>* state_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / c3_cartpole_problem.options.publish_frequency, 4);

  // Connect the simulator's output to the ZeroOrderHold input.
  builder.Connect(lcs_simulator->get_output_port_next_state(),
                  state_zero_order_hold->get_input_port());
  // Connect the ZeroOrderHold output to the simulator's state input.
  builder.Connect(state_zero_order_hold->get_output_port(),
                  lcs_simulator->get_input_port_state());

  // Add cartpole geometry to the builder.
  CartpoleGeometry* geometry = CartpoleGeometry::AddToBuilder(
      &builder, scene_graph, state_zero_order_hold->get_output_port(), 0.0);

  // Add the C3 controller to the builder.
  C3Controller* c3_controller = builder.AddSystem<C3Controller>(
      *(geometry->plant), c3_cartpole_problem.options);
  c3_controller->UpdateCostMatrices(
      *(c3_cartpole_problem.pCost));  // Update cost matrices in the controller.

  // Add a constant value source for the LCS system.
  auto lcs = builder.AddSystem<drake::systems::ConstantValueSource>(
      drake::Value<c3::LCS>(*(c3_cartpole_problem.pSystem)));

  // Add a constant vector source for the desired state.
  drake::systems::ConstantVectorSource<double>* xdes =
      builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
          c3_cartpole_problem.xdesired.at(0));

  Vector2TimestampedVector* vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>();
  // Connect the state output to the timestamped vector system.
  builder.Connect(state_zero_order_hold->get_output_port(),
                  vector_to_timestamped_vector->get_input_port_state());

  // Connect the simulator's output to the controller's state input.
  builder.Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());

  // Connect the LCS system output to the controller's LCS input.
  builder.Connect(lcs->get_output_port(), c3_controller->get_input_port_lcs());

  // Connect the desired state output to the controller's target input.
  builder.Connect(xdes->get_output_port(),
                  c3_controller->get_input_port_target());

  // Add the C3 solution action system to the builder.
  C3Solution2Input* c3_action = builder.AddSystem<C3Solution2Input>();
  builder.Connect(c3_controller->get_output_port_c3_solution(),
                  c3_action->get_input_port_c3_solution());

  // Connect the action output to the simulator's action input.
  builder.Connect(c3_action->get_output_port_c3_action(),
                  lcs_simulator->get_input_port_action());

  // Connect the LCS system output to the simulator's LCS input.
  builder.Connect(lcs->get_output_port(), lcs_simulator->get_input_port_lcs());

  // Set up the Meshcat visualizer.
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizerParams params;
  auto visualizer = &drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, *scene_graph, meshcat, std::move(params));

  // Build the system diagram.
  std::unique_ptr<drake::systems::Diagram<double>> diagram_ = builder.Build();

  // Create a default context for the diagram.
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram_->CreateDefaultContext();

  // Set initial positions for the state subsystem.
  drake::systems::Context<double>& state_context =
      diagram_->GetMutableSubsystemContext(*(state_zero_order_hold),
                                           diagram_context.get());
  Eigen::Vector4d positions = {0, -0.5, 0.5, -0.4};
  state_zero_order_hold->SetVectorState(&state_context, positions);

  // Create and configure the simulator.
  drake::systems::Simulator<double> simulator(*diagram_,
                                              std::move(diagram_context));
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(
      1.0);                // Run simulation at half real-time speed.
  simulator.Initialize();  // Initialize the simulator.

  // Advance the simulation to 10 seconds.
  simulator.AdvanceTo(10.0);

  // Return error code -1 to indicate the end of the program.
  return -1;
}

// Main entry point of the program.
int main(int argc, char** argv) { return DoMain(); }
