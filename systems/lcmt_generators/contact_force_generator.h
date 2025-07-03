#pragma once

#include <string>
#include <vector>

#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

#include "c3/lcmt_contact_forces.hpp"

namespace c3 {
namespace systems {
namespace lcmt_generators {

/**
 * @class ContactForceGenerator
 * @brief Converts solution vectors and LCS contact information into LCM contact
 * force messages for publishing.
 * @details
 *   - Provides input ports for the C3 solution vector and LCS contact
 * information.
 *   - Computes contact forces based on the provided inputs.
 *   - Offers an output port for the constructed contact force message.
 *   - Includes a static helper to add an LCM publisher system to a
 * DiagramBuilder for message transmission.
 */
class ContactForceGenerator : public drake::systems::LeafSystem<double> {
 public:
  /**
   * @brief Constructs a ContactForceGenerator system.
   *
   * Declares input and output ports for the solution vector, LCS contact info,
   * and contact force output.
   */
  ContactForceGenerator();

  /**
   * @brief Returns the input port for the c3 solution vector.
   * @return Reference to the input port for the c3 solution.
   */
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_);
  }

  /**
   * @brief Returns the input port for the LCS contact information.
   * @return Reference to the input port for LCS contact info.
   */
  const drake::systems::InputPort<double>& get_input_port_lcs_contact_info()
      const {
    return this->get_input_port(lcs_contact_info_port_);
  }

  /**
   * @brief Returns the output port for the computed contact forces.
   * @return Reference to the output port for contact forces.
   */
  const drake::systems::OutputPort<double>& get_output_port_contact_force()
      const {
    return this->get_output_port(contact_force_output_port_);
  }

  /**
   * @brief Adds an LCM publisher system to the given DiagramBuilder for
   * publishing contact forces.
   *
   * @param builder The DiagramBuilder to which the publisher will be added.
   * @param solution_port Output port providing the solution vector.
   * @param lcs_contact_info_port Output port providing LCS contact information.
   * @param channel The LCM channel name to publish on.
   * @param lcm The LCM interface to use for publishing.
   * @param publish_triggers Set of triggers that determine when publishing
   * occurs.
   * @param publish_period Optional period for periodic publishing (default:
   * 0.0).
   * @param publish_offset Optional offset for periodic publishing (default:
   * 0.0).
   * @return Pointer to the created LcmPublisherSystem.
   */
  static drake::systems::lcm::LcmPublisherSystem* AddLcmPublisherToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const drake::systems::OutputPort<double>& solution_port,
      const drake::systems::OutputPort<double>& lcs_contact_info_port,
      const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
      const drake::systems::TriggerTypeSet& publish_triggers,
      double publish_period = 0.0, double publish_offset = 0.0);

 private:
  /**
   * @brief Calculates the contact forces and populates the output message.
   *
   * @param context The system context containing input values.
   * @param c3_forces_output Pointer to the output message to populate.
   */
  void DoCalc(const drake::systems::Context<double>& context,
              c3::lcmt_contact_forces* c3_forces_output) const;

  drake::systems::InputPortIndex
      c3_solution_port_;  ///< Index of the c3 solution input port.
  drake::systems::InputPortIndex
      lcs_contact_info_port_;  ///< Index of the LCS contact info input port.
  drake::systems::OutputPortIndex
      contact_force_output_port_;  ///< Index of the contact force output port.
};

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3