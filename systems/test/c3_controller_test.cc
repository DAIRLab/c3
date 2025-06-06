// Includes for core controllers, simulators, and test problems.
#include "systems/c3_controller.h"

#include "common/system_util.hpp"
#include "core/test/c3_cartpole_problem.hpp"
#include "systems/lcs_simulator.h"
#include "systems/test/cartpole_geometry.hpp"

// Includes for Drake systems and primitives.
#include <drake/geometry/drake_visualizer.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/pass_through.h>
#include <drake/systems/primitives/vector_log_sink.h>
#include <drake/systems/primitives/zero_order_hold.h>

using c3::systems::C3Controller;
using c3::systems::C3Output;
using c3::systems::LCSSimulator;
using c3::systems::test::CartpoleWithSoftWallsVisualizer;
using drake::geometry::SceneGraph;
using drake::systems::DiagramBuilder;
using Eigen::VectorXd;

// Custom LeafSystem to process C3 solutions and to get input values.
class C3Solution2Input : public drake::systems::LeafSystem<double> {
 public:
  explicit C3Solution2Input() {
    // Declare input port for C3 solutions.
    c3_solution_port_index_ =
        this->DeclareAbstractInputPort("c3_solution",
                                       drake::Value<C3Output::C3Solution>())
            .get_index();
    // Declare output port for inputs.
    c3_input_port_index_ =
        this->DeclareVectorOutputPort("u", 1, &C3Solution2Input::GetC3Input)
            .get_index();
  }

  // Getter for the input port.
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_index_);
  }

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_output_port_c3_input() const {
    return this->get_output_port(c3_input_port_index_);
  }

 private:
  drake::systems::InputPortIndex c3_solution_port_index_;
  drake::systems::OutputPortIndex c3_input_port_index_;

  // Compute the input from the C3 solution.
  void GetC3Input(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const {
    const drake::AbstractValue* input = this->EvalAbstractInput(context, 0);
    DRAKE_ASSERT(input != nullptr);
    const auto& sol = input->get_value<C3Output::C3Solution>();
    output->SetAtIndex(0, sol.u_sol_(0));  // Set input output.
  }
};

// Converts a vector to a timestamped vector.
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

  // Convert input vector to timestamped vector.
  void Convert(const drake::systems::Context<double>& context,
               c3::systems::TimestampedVector<double>* output) const {
    output->SetDataVector(
        this->EvalVectorInput(context, vector_port_index_)->get_value());
    output->set_timestamp(context.get_time());  // Set timestamp.
  }
};

// Main function to build and run the simulation.
int DoMain() {
  DiagramBuilder<double> builder;
  SceneGraph<double>* scene_graph = builder.AddSystem<SceneGraph>();

  // Initialize the C3 cartpole problem.
  auto c3_cartpole_problem = C3CartpoleProblem();

  // Add the LCS simulator.
  auto lcs_simulator =
      builder.AddSystem<LCSSimulator>(*(c3_cartpole_problem.pSystem));

  // Add a ZeroOrderHold system for state updates.
  auto state_zero_order_hold =
      builder.AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / c3_cartpole_problem.options.publish_frequency, c3_cartpole_problem.n);

  // Connect simulator and ZeroOrderHold.
  builder.Connect(lcs_simulator->get_output_port_next_state(),
                  state_zero_order_hold->get_input_port());
  builder.Connect(state_zero_order_hold->get_output_port(),
                  lcs_simulator->get_input_port_state());

  // Add cartpole geometry.
  CartpoleWithSoftWallsVisualizer* geometry =
      CartpoleWithSoftWallsVisualizer::AddToBuilder(
          &builder, scene_graph, state_zero_order_hold->get_output_port(), 0.0);

  // Add the C3 controller.
  auto c3_controller = builder.AddSystem<C3Controller>(
      *(geometry->plant), c3_cartpole_problem.cost,
      c3_cartpole_problem.options);

  // Add constant value source for the LCS system.
  auto lcs = builder.AddSystem<drake::systems::ConstantValueSource>(
      drake::Value<c3::LCS>(*(c3_cartpole_problem.pSystem)));

  // Add constant vector source for the desired state.
  auto xdes = builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
      c3_cartpole_problem.xdesired.at(0));

  // Add vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      builder.AddSystem<Vector2TimestampedVector>();
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
  auto c3_input = builder.AddSystem<C3Solution2Input>();
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
  simulator.AdvanceTo(10.0);  // Run simulation for 10 seconds.

  return -1;  // Indicate end of program.
}

// Main entry point of the program.
int main(int argc, char** argv) { return DoMain(); }
