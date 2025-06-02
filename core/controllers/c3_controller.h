#pragma once

#include <string>
#include <vector>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_options.h"
#include "core/c3_output.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "core/solver_options_io.h"
#include "systems/framework/timestamped_vector.h"

#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace c3 {
namespace systems {

// C3Controller: A controller for solving C3 optimization problems.
class C3Controller : public drake::systems::LeafSystem<double> {
 public:
  // Constructor: Initializes the controller with the given plant and options.
  explicit C3Controller(const drake::multibody::MultibodyPlant<double>& plant,
                        C3Options c3_options);

  // Accessors for input ports.
  const drake::systems::InputPort<double>& get_input_port_target() const {
    return this->get_input_port(target_input_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_lcs_state() const {
    return this->get_input_port(lcs_state_input_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_lcs() const {
    return this->get_input_port(lcs_input_port_);
  }

  // Accessors for output ports.
  const drake::systems::OutputPort<double>& get_output_port_c3_solution() const {
    return this->get_output_port(c3_solution_port_);
  }
  const drake::systems::OutputPort<double>& get_output_port_c3_intermediates() const {
    return this->get_output_port(c3_intermediates_port_);
  }

  // Sets OSQP solver options.
  void SetOsqpSolverOptions(const drake::solvers::SolverOptions& options) {
    solver_options_ = options;
    c3_->SetOsqpSolverOptions(solver_options_);
  }

 private:
  // Computes the C3 plan and updates discrete state.
  drake::systems::EventStatus ComputePlan(
      const drake::systems::Context<double>& context,
      drake::systems::DiscreteValues<double>* discrete_state) const;

  // Outputs the C3 solution.
  void OutputC3Solution(const drake::systems::Context<double>& context,
                        C3Output::C3Solution* c3_solution) const;

  // Outputs intermediate C3 results.
  void OutputC3Intermediates(const drake::systems::Context<double>& context,
                             C3Output::C3Intermediates* c3_intermediates) const;

  // Input and output port indices.
  drake::systems::InputPortIndex target_input_port_;
  drake::systems::InputPortIndex lcs_state_input_port_;
  drake::systems::InputPortIndex lcs_input_port_;
  drake::systems::OutputPortIndex c3_solution_port_;
  drake::systems::OutputPortIndex c3_intermediates_port_;

  // Reference to the multibody plant.
  const drake::multibody::MultibodyPlant<double>& plant_;

  // C3 options and solver configuration.
  C3Options c3_options_;
  drake::solvers::SolverOptions solver_options_ =
      drake::yaml::LoadYamlFile<c3::SolverOptionsFromYaml>(
          "core/configs/solver_options_default.yaml")
          .GetAsSolverOptions(drake::solvers::OsqpSolver::id());

  // Convenience variables for dimensions.
  int n_q_;       // Number of generalized positions.
  int n_v_;       // Number of generalized velocities.
  int n_x_;       // Total state dimension.
  int n_lambda_;  // Number of Lagrange multipliers.
  int n_u_;       // Number of control inputs.
  double dt_;     // Time step.

  // C3 solver instance.
  mutable std::unique_ptr<C3> c3_;

  // Solve time filter constant.
  double solve_time_filter_constant_;

  // Indices for discrete state variables.
  drake::systems::DiscreteStateIndex plan_start_time_index_;
  drake::systems::DiscreteStateIndex x_pred_index_;
  drake::systems::DiscreteStateIndex filtered_solve_time_index_;

  // Cost matrices for optimization.
  std::vector<Eigen::MatrixXd> Q_;  // State cost matrices.
  std::vector<Eigen::MatrixXd> R_;  // Input cost matrices.
  std::vector<Eigen::MatrixXd> G_;  // State-input cross-term matrices.
  std::vector<Eigen::MatrixXd> U_;  // Constraint matrices.

  int N_;  // Horizon length.
};

}  // namespace systems
}  // namespace c3
