#pragma once

#include <string>
#include <vector>

#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

#include "c3/lcmt_output.hpp"

namespace c3 {
namespace systems {
namespace publishers {
/// Converts a OutputVector object to LCM type lcmt_robot_output
class C3OutputPublisher : public drake::systems::LeafSystem<double> {
 public:
  C3OutputPublisher();

  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_);
  }

  const drake::systems::InputPort<double>& get_input_port_c3_intermediates()
      const {
    return this->get_input_port(c3_intermediates_port_);
  }

  const drake::systems::OutputPort<double>& get_output_port_c3_output() const {
    return this->get_output_port(c3_output_port_);
  }

  static drake::systems::lcm::LcmPublisherSystem* AddLcmPublisherToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const drake::systems::OutputPort<double>& solution_port,
      const drake::systems::OutputPort<double>& intermediates_port,
      const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
      const drake::systems::TriggerTypeSet& publish_triggers,
      double publish_period = 0.0, double publish_offset = 0.0);

 private:
  void DoCalc(const drake::systems::Context<double>& context,
              c3::lcmt_output* output) const;

  //  void OutputNextC3Input(const drake::systems::Context<double>& context,
  //                         drake::systems::BasicVector<double>* u_next) const;

  drake::systems::InputPortIndex c3_solution_port_;
  drake::systems::InputPortIndex c3_intermediates_port_;
  drake::systems::OutputPortIndex c3_output_port_;
};
}  // namespace publishers
}  // namespace systems
}  // namespace c3