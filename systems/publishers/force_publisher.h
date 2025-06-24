#pragma once

#include <string>
#include <vector>

#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

#include "c3/lcmt_forces.hpp"

namespace c3 {
namespace systems {
namespace publishers {
class ContactForcePublisher : public drake::systems::LeafSystem<double> {
 public:
  ContactForcePublisher();

  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_);
  }

  const drake::systems::InputPort<double>& get_input_port_lcs_contact_info()
      const {
    return this->get_input_port(lcs_contact_info_port_);
  }

  // LCS contact force, not actor input forces
  const drake::systems::OutputPort<double>& get_output_port_contact_force()
      const {
    return this->get_output_port(contact_force_output_port_);
  }

  static drake::systems::lcm::LcmPublisherSystem*
  AddLcmPublisherToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const drake::systems::OutputPort<double>& solution_port,
      const drake::systems::OutputPort<double>& lcs_contact_info_port,
      const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
      const drake::systems::TriggerTypeSet& publish_triggers,
      double publish_period = 0.0, double publish_offset = 0.0);

 private:
  void DoCalc(const drake::systems::Context<double>& context,
              c3::lcmt_forces* c3_forces_output) const;

  drake::systems::InputPortIndex c3_solution_port_;
  drake::systems::InputPortIndex lcs_contact_info_port_;
  drake::systems::OutputPortIndex contact_force_output_port_;
};

}  // namespace publishers
}  // namespace systems
}  // namespace c3