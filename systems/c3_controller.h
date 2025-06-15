#pragma once

#include <string>
#include <vector>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_options.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

namespace c3 {
namespace systems {

/**
 * @class C3Controller
 * @brief A controller for solving C3 optimization problems.
 *
 * This class implements a controller that uses the C3 optimization framework
 * to compute control inputs for a multibody system. It provides methods for
 * setting solver options, updating cost matrices, and adding constraints.
 */
class C3Controller : public drake::systems::LeafSystem<double> {
 public:
  /**
   * @brief Constructor: Initializes the controller with the given plant and
   * options.
   * @param plant The multibody plant to control.
   * @param costs The cost matrices for the C3 optimization problem.
   * @param controller_options Options for configuring the C3 controller and
   * solver.
   */
  explicit C3Controller(const drake::multibody::MultibodyPlant<double>& plant,
                        const C3::CostMatrices& costs,
                        C3ControllerOptions controller_options);

  // Accessors for input ports.
  const drake::systems::InputPort<double>& get_input_port_target() const {
    return this->get_input_port(lcs_desired_state_input_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_lcs_state() const {
    return this->get_input_port(lcs_state_input_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_lcs() const {
    return this->get_input_port(lcs_input_port_);
  }

  // Accessors for output ports.
  const drake::systems::OutputPort<double>& get_output_port_c3_solution()
      const {
    return this->get_output_port(c3_solution_port_);
  }
  const drake::systems::OutputPort<double>& get_output_port_c3_intermediates()
      const {
    return this->get_output_port(c3_intermediates_port_);
  }

  /**
   * @brief Updates the cost matrices used by the controller.
   * @param costs The new cost matrices to be used in the optimization problem.
   */
  void UpdateCostMatrices(C3::CostMatrices& costs) {
    c3_->UpdateCostMatrices(costs);
  }

  /**
   * @brief Adds a linear constraint to the controller.
   * @param A The constraint matrix.
   * @param lower_bound The lower bound of the constraint.
   * @param upper_bound The upper bound of the constraint.
   * @param constraint The type of constraint variable.
   */
  void AddLinearConstraint(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& lower_bound,
                           const Eigen::VectorXd& upper_bound,
                           enum ConstraintVariable constraint) {
    c3_->AddLinearConstraint(A, lower_bound, upper_bound, constraint);
  }

  /**
   * @brief Adds a linear constraint to the controller (row vector version).
   * @param A The constraint row vector.
   * @param lower_bound The lower bound of the constraint.
   * @param upper_bound The upper bound of the constraint.
   * @param constraint The type of constraint variable.
   */
  void AddLinearConstraint(const Eigen::RowVectorXd& A, double lower_bound,
                           double upper_bound,
                           enum ConstraintVariable constraint) {
    c3_->AddLinearConstraint(A, lower_bound, upper_bound, constraint);
  }

 private:
  /**
   * @struct StatePredictionJoint
   * @brief Represents the configuration and constraints for a joint in a state
   * prediction system.
   *
   * This structure is used to define the indices and sizes of position (q) and
   * velocity (v) components for a joint, as well as the maximum allowable
   * acceleration for the joint.
   */
  struct StatePredictionJoint {
    int q_start_index;        ///< Starting index of the position (q) component.
    int q_size;               ///< Size of the position (q) component.
    int v_start_index;        ///< Starting index of the velocity (v) component.
    int v_size;               ///< Size of the velocity (v) component.
    double max_acceleration;  ///< Maximum allowable acceleration for the joint.
  };

  /**
   * @brief Adjusts the input state vector by clamping it to a predicted state
   * based on maximum acceleration constraints for specified joints.
   *
   * This function modifies the provided state vector `x0` by incorporating
   * a predicted state derived from the previous C3 solution. The predicted
   * state is interpolated based on the filtered solve time and clamped to
   * ensure that joint positions and velocities do not exceed the maximum
   * acceleration constraints defined for the state prediction joints. If the
   * solve time is zero, the function returns early without making any
   * modifications.
   *
   * @param x0 A reference to the state vector to be adjusted. This vector
   *           contains the positions and velocities of the system.
   */
  void ClampPredictedState(drake::VectorX<double>& x0,
                           double& filtered_solve_time) const;

  /**
   * @brief Computes the C3 solution and intermediates given a discrete state.
   * @param context The system context.
   * @param discrete_state The discrete state (usually the current state of the
   * system).
   * @return The event status after computation.
   */
  drake::systems::EventStatus ComputePlan(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  /**
   * @brief Outputs the C3 solution.
   * @param context The system context.
   * @param c3_solution The C3 solution to output.
   */
  void OutputC3Solution(const drake::systems::Context<double>& context,
                        C3Output::C3Solution* c3_solution) const;

  /**
   * @brief Outputs intermediate C3 results.
   * @param context The system context.
   * @param c3_intermediates The intermediate results to output.
   */
  void OutputC3Intermediates(const drake::systems::Context<double>& context,
                             C3Output::C3Intermediates* c3_intermediates) const;

  // Input and output port indices.
  drake::systems::InputPortIndex lcs_desired_state_input_port_;
  drake::systems::InputPortIndex lcs_state_input_port_;
  drake::systems::InputPortIndex lcs_input_port_;
  drake::systems::OutputPortIndex c3_solution_port_;
  drake::systems::OutputPortIndex c3_intermediates_port_;

  // Reference to the multibody plant.
  const drake::multibody::MultibodyPlant<double>& plant_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  double publish_frequency_;

  // Convenience variables for dimensions.
  int n_q_;       ///< Number of generalized positions.
  int n_v_;       ///< Number of generalized velocities.
  int n_x_;       ///< Total state dimension.
  int n_lambda_;  ///< Number of Lagrange multipliers.
  int n_u_;       ///< Number of control inputs.
  double dt_;     ///< Time step.

  // C3 solver instance.
  mutable std::unique_ptr<C3> c3_;

  // Solve time filter constant.
  double solve_time_filter_constant_;

  // mutable values.
  drake::systems::DiscreteStateIndex plan_start_time_index_;
  drake::systems::DiscreteStateIndex filtered_solve_time_index_;

  std::vector<StatePredictionJoint> state_prediction_joints_;

  // Cost matrices for optimization.
  std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.

  int N_;  ///< Horizon length.
};

}  // namespace systems
}  // namespace c3
