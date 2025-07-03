#include "trajectory_generator.h"

namespace c3 {
namespace systems {
namespace lcmt_generators {

TrajectoryGenerator::TrajectoryGenerator(TrajectoryGeneratorOptions options)
    : options_(options) {
  // Declare input ports for solution and intermediates.
  c3_solution_input_port_ = this->DeclareAbstractInputPort(
      "c3_solution", drake::Value<C3Output::C3Solution>()).get_index();
  c3_intermediates_input_port_ = this->DeclareAbstractInputPort(
      "c3_intermediates", drake::Value<C3Output::C3Intermediates>()).get_index();

  // Declare output port for lcmt_trajectory message.
  lcmt_trajectory_output_port_ = this->DeclareAbstractOutputPort(
      "lcmt_trajectory", &TrajectoryGenerator::GenerateTrajectory).get_index();

  this->set_name("c3_trajectory_generator");
}

void TrajectoryGenerator::GenerateTrajectory(
    const drake::systems::Context<double>& context,
    lcmt_trajectory* output) const {
  // Get input data
  const auto* solution = this->EvalInputValue<C3Output::C3Solution>(
      context, c3_solution_input_port_);
  const auto* intermediates = this->EvalInputValue<C3Output::C3Intermediates>(
      context, c3_intermediates_input_port_);

  // Example: Fill in the output message with dummy data for demonstration.
  // In practice, use options_ and input data to populate the message.
  output->trajectory_name = "example_trajectory";
  output->num_timesteps = 10;
  output->vector_dim = 2;
  output->timestamps.resize(10);
  output->values.resize(2, std::vector<double>(10, 0.0));
  for (int i = 0; i < 10; ++i) {
    output->timestamps[i] = 0.1 * i;
    output->values[0][i] = i;
    output->values[1][i] = 2 * i;
  }
}

drake::systems::lcm::LcmPublisherSystem* TrajectoryGenerator::AddLcmPublisherToBuilder(
    drake::systems::DiagramBuilder<double>& builder,
    const drake::systems::OutputPort<double>& c3_solution_port,
    const drake::systems::OutputPort<double>& c3_intermediates_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  auto traj_gen = builder.AddSystem<TrajectoryGenerator>(TrajectoryGeneratorOptions{});
  builder.Connect(c3_solution_port, traj_gen->get_input_port_c3_solution());
  builder.Connect(c3_intermediates_port, traj_gen->get_input_port_c3_intermediates());

  auto lcm_pub = builder.AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<lcmt_trajectory>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(traj_gen->get_output_port_lcmt_trajectory(), lcm_pub->get_input_port());
  return lcm_pub;
}

}  // namespace lcmt_generators
}  // namespace systems
}  //