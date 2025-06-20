#pragma once

#include <string>
#include <vector>

#include <drake/multibody/plant/multibody_plant.h>

#include "core/c3_options.h"
#include "core/lcs.h"
#include "multibody/lcs_factory.h"

#include "drake/systems/framework/leaf_system.h"

/**
 * @file lcs_factory_system.h
 * @brief Defines the LCSFactorySystem class, which is responsible for creating
 *        and managing Linear Complementarity Systems (LCS) based on the
 *        change in the MultibodyPlant dynamics and contact geometry.
 */

namespace c3 {
namespace systems {

/**
 * @class LCSFactorySystem
 * @brief A Drake LeafSystem that generates Linear Complementarity Systems (LCS)
 *        from a MultibodyPlant and its associated context.
 *
 * This system provides input ports for LCS state and inputs, and output ports
 * for the generated LCS and its contact Jacobian. It uses the LCSFactory to
 * linearize the dynamics of the MultibodyPlant and compute contact-related
 * properties.
 */
class LCSFactorySystem : public drake::systems::LeafSystem<double> {
 public:
  /**
   * @brief Constructs an LCSFactorySystem.
   *
   * @param plant The MultibodyPlant<double> representing the system dynamics.
   * @param context The Context<double> associated with the plant.
   * @param plant_ad The MultibodyPlant<drake::AutoDiffXd> for automatic
   * differentiation.
   * @param context_ad The Context<drake::AutoDiffXd> associated with the
   * plant_ad.
   * @param contact_geoms A vector of contact geometry pairs used for contact
   * calculations.
   * @param options Configuration options for the LCSFactory.
   */
  explicit LCSFactorySystem(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>
          contact_geoms,
      LCSOptions options);

  /**
   * @brief Gets the input port for the LCS state.
   *
   * @return A reference to the input port for the LCS state.
   */
  const drake::systems::InputPort<double>& get_input_port_lcs_state() const {
    return this->get_input_port(lcs_state_input_port_);
  }

  /**
   * @brief Gets the input port for the LCS inputs.
   *
   * @return A reference to the input port for the LCS inputs.
   */
  const drake::systems::InputPort<double>& get_input_port_lcs_input() const {
    return this->get_input_port(lcs_inputs_input_port_);
  }

  /**
   * @brief Gets the output port for the generated LCS.
   *
   * @return A reference to the output port for the LCS.
   */
  const drake::systems::OutputPort<double>& get_output_port_lcs() const {
    return this->get_output_port(lcs_port_);
  }

  /**
   * @brief Gets the output port for the LCS contact Jacobian.
   *
   * @return A reference to the output port for the LCS contact Jacobian.
   */
  const drake::systems::OutputPort<double>&
  get_output_port_lcs_contact_jacobian() const {
    return this->get_output_port(lcs_contact_jacobian_port_);
  }

  static LCSFactorySystem* AddToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const drake::multibody::MultibodyPlant<double>& plant_for_lcs,
      drake::systems::Context<double>* plant_diagram_context,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>
          contact_geoms,
      LCSOptions options);

 private:
  /**
   * @brief Computes the LCS based on the current state and inputs.
   *
   * @param context The system context containing the current state and inputs.
   * @param output_traj Pointer to the output LCS object.
   */
  void OutputLCS(const drake::systems::Context<double>& context,
                 LCS* output_lcs) const;

  /**
   * @brief Computes the contact Jacobian and contact points for the LCS.
   *
   * @param context The system context containing the current state.
   * @param output Pointer to the output pair containing the contact Jacobian
   *               and contact points.
   */
  void OutputLCSContactJacobian(
      const drake::systems::Context<double>& context,
      std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>* output) const;

  // Member variables for input and output port indices
  drake::systems::InputPortIndex lcs_state_input_port_;
  drake::systems::InputPortIndex lcs_inputs_input_port_;
  drake::systems::OutputPortIndex lcs_port_;
  drake::systems::OutputPortIndex lcs_contact_jacobian_port_;

  // Convenience variables for system dimensions
  int n_q_;       ///< Number of positions in the plant.
  int n_v_;       ///< Number of velocities in the plant.
  int n_x_;       ///< Total number of state variables (positions + velocities).
  int n_lambda_;  ///< Number of contact variables.
  int n_u_;       ///< Number of actuators in the plant.
  int N_;         ///< Number of time steps for the LCS.
  double dt_;     ///< Time step size for the LCS.

  std::unique_ptr<multibody::LCSFactory> lcs_factory_;  ///< Factory for creating LCS objects.
};

}  // namespace systems
}  // namespace c3
