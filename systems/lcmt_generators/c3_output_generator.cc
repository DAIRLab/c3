#include "systems/lcmt_generators/c3_output_generator.h"

#include "systems/framework/c3_output.h"

using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::lcm::LcmPublisherSystem;

namespace c3 {
namespace systems {
namespace lcmt_generators {

// Publishes C3Output as an LCM message.
C3OutputGenerator::C3OutputGenerator() {
  // Declare input port for the C3 solution.
  c3_solution_port_ =
      this->DeclareAbstractInputPort("c3_solution",
                                     drake::Value<C3Output::C3Solution>{})
          .get_index();
  // Declare input port for C3 intermediates.
  c3_intermediates_port_ =
      this->DeclareAbstractInputPort("c3_intermediates",
                                     drake::Value<C3Output::C3Intermediates>{})
          .get_index();

  this->set_name("c3_output_publisher");
  // Declare output port for the LCM message.
  c3_output_port_ =
      this->DeclareAbstractOutputPort("lcmt_c3_output", c3::lcmt_output(),
                                      &C3OutputGenerator::DoCalc)
          .get_index();
}

// Calculates the LCM output message from the input ports.
void C3OutputGenerator::DoCalc(const drake::systems::Context<double>& context,
                               c3::lcmt_output* output) const {
  // Retrieve input values.
  const auto& c3_solution =
      this->EvalInputValue<C3Output::C3Solution>(context, c3_solution_port_);
  const auto& c3_intermediates =
      this->EvalInputValue<C3Output::C3Intermediates>(context,
                                                      c3_intermediates_port_);

  // Construct C3Output and generate the LCM message.
  C3Output c3_output = C3Output(*c3_solution, *c3_intermediates);
  *output = c3_output.GenerateLcmObject(context.get_time());
}

// Adds this publisher and an LCM publisher system to the diagram builder.
LcmPublisherSystem* C3OutputGenerator::AddLcmPublisherToBuilder(
    DiagramBuilder<double>& builder,
    const drake::systems::OutputPort<double>& solution_port,
    const drake::systems::OutputPort<double>& intermediates_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  // Add and connect the C3OutputGenerator system.
  auto output_publisher = builder.AddSystem<C3OutputGenerator>();
  builder.Connect(solution_port,
                  output_publisher->get_input_port_c3_solution());
  builder.Connect(intermediates_port,
                  output_publisher->get_input_port_c3_intermediates());

  // Add and connect the LCM publisher system.
  auto lcm_output_publisher =
      builder.AddSystem(LcmPublisherSystem::Make<c3::lcmt_output>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(output_publisher->get_output_port_c3_output(),
                  lcm_output_publisher->get_input_port());
  return lcm_output_publisher;
}

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3