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
namespace lcmt_generators {

/**
 * @class C3OutputGenerator
 * @brief Converts OutputVector objects to LCM type lcmt_output for publishing.
 * @details
 *   - Provides input ports for C3 solution and intermediates.
 *   - Provides an output port for the constructed lcmt_output message.
 *   - Can be connected to an LCM publisher system for message transmission.
 */
class C3OutputGenerator : public drake::systems::LeafSystem<double> {
 public:
  /**
   * @brief Constructor. Declares input and output ports.
   */
  C3OutputGenerator();

  /**
   * @brief Returns the input port for the C3 solution vector.
   */
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_);
  }

  /**
   * @brief Returns the input port for the C3 intermediates vector.
   */
  const drake::systems::InputPort<double>& get_input_port_c3_intermediates()
      const {
    return this->get_input_port(c3_intermediates_port_);
  }

  /**
   * @brief Returns the output port for the lcmt_output message.
   */
  const drake::systems::OutputPort<double>& get_output_port_c3_output() const {
    return this->get_output_port(c3_output_port_);
  }

  /**
   * @brief Adds an LCM publisher system to the given DiagramBuilder.
   * @param builder The diagram builder to add the publisher to.
   * @param solution_port Output port for the solution vector.
   * @param intermediates_port Output port for the intermediates vector.
   * @param channel LCM channel name.
   * @param lcm LCM interface pointer.
   * @param publish_triggers Set of triggers for publishing.
   * @param publish_period Period for periodic publishing (optional).
   * @param publish_offset Offset for periodic publishing (optional).
   * @return Pointer to the created LcmPublisherSystem.
   */
  static drake::systems::lcm::LcmPublisherSystem* AddLcmPublisherToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const drake::systems::OutputPort<double>& solution_port,
      const drake::systems::OutputPort<double>& intermediates_port,
      const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
      const drake::systems::TriggerTypeSet& publish_triggers,
      double publish_period = 0.0, double publish_offset = 0.0);

 private:
  /**
   * @brief Calculates the lcmt_output message from the input ports.
   * @param context The system context.
   * @param output The output message to populate.
   */
  void DoCalc(const drake::systems::Context<double>& context,
              c3::lcmt_output* output) const;

  drake::systems::InputPortIndex
      c3_solution_port_; /**< Index for solution input port. */
  drake::systems::InputPortIndex
      c3_intermediates_port_; /**< Index for intermediates input port. */
  drake::systems::OutputPortIndex
      c3_output_port_; /**< Index for output port. */
};

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3