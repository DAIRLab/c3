#include "output_publisher.h"

#include "systems/framework/c3_output.h"

using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::lcm::LcmPublisherSystem;

namespace c3 {
namespace systems {
namespace publishers {

C3OutputPublisher::C3OutputPublisher() {
  c3_solution_port_ =
      this->DeclareAbstractInputPort("c3_solution",
                                     drake::Value<C3Output::C3Solution>{})
          .get_index();
  c3_intermediates_port_ =
      this->DeclareAbstractInputPort("c3_intermediates",
                                     drake::Value<C3Output::C3Intermediates>{})
          .get_index();

  this->set_name("c3_output_publisher");
  c3_output_port_ =
      this->DeclareAbstractOutputPort("lcmt_c3_output", c3::lcmt_output(),
                                      &C3OutputPublisher::DoCalc)
          .get_index();
}

void C3OutputPublisher::DoCalc(const drake::systems::Context<double>& context,
                               c3::lcmt_output* output) const {
  const auto& c3_solution =
      this->EvalInputValue<C3Output::C3Solution>(context, c3_solution_port_);
  const auto& c3_intermediates =
      this->EvalInputValue<C3Output::C3Intermediates>(context,
                                                      c3_intermediates_port_);

  C3Output c3_output = C3Output(*c3_solution, *c3_intermediates);
  *output = c3_output.GenerateLcmObject(context.get_time());
}

LcmPublisherSystem* C3OutputPublisher::AddLcmPublisherToBuilder(
    DiagramBuilder<double>& builder,
    const drake::systems::OutputPort<double>& solution_port,
    const drake::systems::OutputPort<double>& intermediates_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  auto output_publisher = builder.AddSystem<C3OutputPublisher>();
  builder.Connect(solution_port,
                  output_publisher->get_input_port_c3_solution());
  builder.Connect(intermediates_port,
                  output_publisher->get_input_port_c3_intermediates());

  auto lcm_output_publisher =
      builder.AddSystem(LcmPublisherSystem::Make<c3::lcmt_output>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(output_publisher->get_output_port_c3_output(),
                  lcm_output_publisher->get_input_port());
  return lcm_output_publisher;
}

}  // namespace publishers
}  // namespace systems
}  // namespace c3