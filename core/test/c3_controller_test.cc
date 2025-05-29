#include "core/controllers/c3_controller.h"
#include "core/controllers/lcs_simulator.h"
#include "core/test/c3_cartpole_problem.h"
#include "core/test/cartpole_geometry.hpp"

#include "drake/systems/primitives/constant_value_source.h"
#include "drake/systems/primitives/constant_vector_source.h"

#include "drake/geometry/drake_visualizer.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/meshcat_visualizer_params.h"
#include "drake/systems/analysis/simulator.h"

using drake::geometry::SceneGraph;
using drake::systems::DiagramBuilder;

using c3::systems::C3Controller;
using c3::systems::LCSSimulator;
using c3::test::CartpoleGeometry;

using Eigen::VectorXd;

class C3SolutionAction : public drake::systems::LeafSystem<double> {
public:
  explicit C3SolutionAction() {
    c3_solution_port_index_ =
        this->DeclareAbstractInputPort("c3_solution",
                                       drake::Value<C3Output::C3Solution>())
            .get_index();
    c3_action_port_index_ =
        this->DeclareVectorOutputPort("u", 1, &C3SolutionAction::GetC3Action)
            .get_index();
  }
  const drake::systems::InputPort<double> &get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_index_);
  }
  const drake::systems::OutputPort<double> &get_output_port_c3_action() const {
    return this->get_output_port(c3_action_port_index_);
  }

private:
  drake::systems::InputPortIndex c3_solution_port_index_;
  drake::systems::OutputPortIndex c3_action_port_index_;

  void GetC3Action(const drake::systems::Context<double> &context,
                   drake::systems::BasicVector<double> *output) const {
    // Get Input
    const drake::AbstractValue *input = this->EvalAbstractInput(context, 0);
    DRAKE_ASSERT(input != nullptr);
    const auto &sol = input->get_value<C3Output::C3Solution>();
    output->SetAtIndex(0, sol.u_sol_(0));
  }
};

int DoMain() {
  DiagramBuilder<double> builder;
  SceneGraph<double> *scene_graph = builder.AddSystem<SceneGraph>();

  C3CartpoleProblem c3_cartpole_problem = C3CartpoleProblem();

  LCSSimulator *lcs_simulator =
      builder.AddSystem<LCSSimulator>(c3_cartpole_problem.pSystem.get());
  CartpoleGeometry *geometry = CartpoleGeometry::AddToBuilder(
      &builder, scene_graph, lcs_simulator->get_output_port_next_state(), 0.01);
  c3::systems::C3Controller *c3_controller =
      builder.AddSystem<c3::systems::C3Controller>(*(geometry->plant),
                                                   c3_cartpole_problem.options);

  // auto lcs =
  //     builder.AddSystem<drake::systems::ConstantValueSource>(
  //         drake::Value<c3::LCS>(*(c3_cartpole_problem.pSystem)));
  // drake::systems::ConstantVectorSource<double> *xdes =
  //     builder.AddSystem<drake::systems::ConstantVectorSource<double>>(
  //         c3_cartpole_problem.xdesired.at(0));

  // builder.Connect(lcs_simulator->get_output_port_next_state(),
  //                 c3_controller->get_input_port_lcs_state());
  // builder.Connect(lcs->get_output_port(),
  // c3_controller->get_input_port_lcs());
  // builder.Connect(xdes->get_output_port(),
  //                 c3_controller->get_input_port_target());

  //   C3SolutionAction *c3_action = builder.AddSystem<C3SolutionAction>();
  //   builder.Connect(c3_controller->get_output_port_c3_solution(),
  //                   c3_action->get_input_port_c3_solution());

  //   builder.Connect(c3_action->get_output_port_c3_action(),
  //                   lcs_simulator->get_input_port_action());
  //   builder.Connect(lcs->get_output_port(),
  //   lcs_simulator->get_input_port_lcs());
  //   builder.Connect(lcs_simulator->get_output_port_next_state(),
  //                   lcs_simulator->get_input_port_state());

  //   auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  //   drake::geometry::MeshcatVisualizerParams params;
  //   //   params.publish_period = 1.0 / sim_params.visualizer_publish_rate;
  //   auto visualizer =
  //   &drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
  //       &builder, *scene_graph, meshcat, std::move(params));

  //   std::unique_ptr<drake::systems::Diagram<double>> diagram_ =
  //   builder.Build();

  //   std::unique_ptr<drake::systems::Context<double>> diagram_context =
  //       diagram_->CreateDefaultContext();

  //   // Create plant_context to set velocity.
  //   drake::systems::Context<double> &plant_context =
  //       diagram_->GetMutableSubsystemContext(*(geometry->plant),
  //       diagram_context.get());
  //   // Set init position.
  //   VectorXd positions = drake::Vector2<double>(0.001, 0.0);
  //   geometry->plant->SetPositions(&plant_context, positions);

  //   drake::systems::Simulator<double> simulator(*diagram_,
  //                                               std::move(diagram_context));

  //   simulator.set_target_realtime_rate(0.5);
  //   simulator.Initialize();
  //   simulator.AdvanceTo(10.0);

  return -1;
}

int main(int argc, char **argv) { return DoMain(); }
