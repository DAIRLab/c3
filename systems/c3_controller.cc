#include "c3_controller.h"

#include <cmath>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_qp.h"

namespace c3 {

using drake::multibody::ModelInstanceIndex;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;
using Eigen::VectorXf;
using std::vector;
using systems::TimestampedVector;

namespace systems {

C3Controller::C3Controller(
    const drake::multibody::MultibodyPlant<double>& plant,
    const C3::CostMatrices& costs, C3ControllerOptions controller_options)
    : plant_(plant),
      controller_options_(controller_options),
      publish_frequency_(controller_options.publish_frequency),
      N_(controller_options_.N) {
  this->set_name("c3_controller");

  // Initialize dimensions
  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_u_ = plant_.num_actuators();
  n_x_ = n_q_ + n_v_;
  dt_ = controller_options_.dt;
  solve_time_filter_constant_ = controller_options_.solve_time_filter_alpha;

  for (auto joint_description : controller_options_.state_prediction_joints) {
    const auto joint = &plant.GetJointByName(joint_description.name);
    state_prediction_joints_.push_back(
        StatePredictionJoint(joint->position_start(), joint->num_positions(),
                             joint->velocity_start(), joint->num_velocities(),
                             joint_description.max_acceleration));
  }

  // Determine the size of lambda based on the contact model
  n_lambda_ = 0;
  if (controller_options_.contact_model == "stewart_and_trinkle") {
    n_lambda_ = 2 * controller_options_.num_contacts +
                2 * controller_options_.num_friction_directions *
                    controller_options_.num_contacts;
  } else if (controller_options_.contact_model == "anitescu") {
    n_lambda_ = 2 * controller_options_.num_friction_directions *
                controller_options_.num_contacts;
  } else {
    drake::log()->error("Unknown contact model: {}",
                        controller_options_.contact_model);
  }
  DRAKE_THROW_UNLESS(n_lambda_ > 0);

  // Placeholder vector for initialization
  VectorXd zeros = VectorXd::Zero(n_x_ + n_lambda_ + n_u_);

  // Create placeholder LCS and desired state for the base C3 problem
  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, N_, dt_);
  auto x_desired_placeholder =
      std::vector<VectorXd>(N_ + 1, VectorXd::Zero(n_x_));

  // Initialize the C3 problem based on the projection type
  if (controller_options_.projection_type == "MIQP") {
    c3_ = std::make_unique<C3MIQP>(lcs_placeholder, costs,
                                   x_desired_placeholder, controller_options_);
  } else if (controller_options_.projection_type == "QP") {
    c3_ = std::make_unique<C3QP>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_);
  } else {
    drake::log()->error("Unknown projection type : {}",
                        controller_options_.projection_type);
  }
  DRAKE_THROW_UNLESS(c3_ != nullptr);

  // Declare input ports
  lcs_state_input_port_ =
      this->DeclareVectorInputPort("state", TimestampedVector<double>(n_x_))
          .get_index();
  lcs_input_port_ =
      this->DeclareAbstractInputPort("lcs", drake::Value<LCS>(lcs_placeholder))
          .get_index();
  lcs_desired_state_input_port_ =
      this->DeclareVectorInputPort("desired_state", n_x_).get_index();

  // Declare output ports
  c3_solution_port_ =
      this->DeclareAbstractOutputPort(
              "solution", C3Output::C3Solution(n_x_, n_lambda_, n_u_, N_),
              &C3Controller::OutputC3Solution)
          .get_index();
  c3_intermediates_port_ =
      this->DeclareAbstractOutputPort(
              "intermediates",
              C3Output::C3Intermediates(n_x_, n_lambda_, n_u_, N_),
              &C3Controller::OutputC3Intermediates)
          .get_index();

  // Declare discrete states
  input_state_timestamp_ = 0.0;
  filtered_solve_time_ = 0.0;

  // Declare periodic or forced update events
  if (publish_frequency_ > 0) {
    DeclarePeriodicDiscreteUpdateEvent(1 / publish_frequency_, 0.0,
                                       &C3Controller::ComputePlan);
  } else {
    DeclareForcedDiscreteUpdateEvent(&C3Controller::ComputePlan);
  }
}

drake::systems::EventStatus C3Controller::ComputePlan(
    const Context<double>& context,
    DiscreteValues<double>* discrete_state) const {
  // Start measuring time for the C3 solve
  auto start = std::chrono::high_resolution_clock::now();

  // Retrieve inputs
  // Get the state of the system from the input port
  if (!get_input_port_lcs().HasValue(context)) {
    throw std::runtime_error("Input port LCS [C3Controller] not connected");
  }
  const auto state = (TimestampedVector<double>*)this->EvalVectorInput(
      context, lcs_state_input_port_);
  drake::VectorX<double> x0 = state->get_data();
  input_state_timestamp_ = state->get_timestamp();

  // Get the LCS from the input port
  const auto& lcs =
      this->EvalAbstractInput(context, lcs_input_port_)->get_value<LCS>();
  // Get the desired state from the input port
  const auto& desired_state = *this->EvalVectorInput<BasicVector>(
      context, lcs_desired_state_input_port_);
  std::vector<VectorXd> target =
      std::vector<VectorXd>(N_ + 1, desired_state.value());

  if (!state_prediction_joints_.empty()) UseClampedPredictedState(x0);

  // Update LCS and target in the C3 problem
  c3_->UpdateLCS(lcs);
  c3_->UpdateTarget(target);
  c3_->Solve(x0);

  // Measure solve time
  auto finish = std::chrono::high_resolution_clock::now();
  double time_ellapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(finish - start)
          .count() /
      1e6;

  // Update solve time in the discrete state
  if (publish_frequency_ > 0) {
    // Override solve time if a publish frequency is specified
    filtered_solve_time_ = 1.0 / publish_frequency_;
  } else {
    // Update to solve time using a moving average filter
    filtered_solve_time_ = (1 - solve_time_filter_constant_) * time_ellapsed +
                           solve_time_filter_constant_ * filtered_solve_time_;
  }

  return drake::systems::EventStatus::Succeeded();
}

void C3Controller::UseClampedPredictedState(drake::VectorX<double>& x0) const {
  // Return early if solve time is zero, as clamping cannot be performed
  if (filtered_solve_time_ == 0.0) {
    return;
  }

  // Retrieve the predicted state from the C3 solution
  std::vector<Eigen::VectorXd> x_sol = c3_->GetStateSolution();
  Eigen::VectorXd predicted_state = x_sol[N_ - 1];

  // Interpolate the predicted state if the filtered solve time is within bounds
  if (filtered_solve_time_ < (N_ - 1) * dt_) {
    int predicted_state_index = std::floor(filtered_solve_time_ / dt_);
    double averaging_weight =
        filtered_solve_time_ / dt_ - predicted_state_index;
    predicted_state = (1 - averaging_weight) * x_sol[predicted_state_index] +
                      averaging_weight * x_sol[predicted_state_index + 1];
  }

  // Approximate the loop time step for clamping
  float approx_loop_dt = std::min(dt_, filtered_solve_time_);

  // Clamp the predicted state based on maximum acceleration constraints
  for (const StatePredictionJoint& joint : state_prediction_joints_) {
    for (int idx = joint.q_start; idx < joint.q_start + joint.q_size; ++idx) {
      // Ensure the joint indices are within bounds
      DRAKE_DEMAND(idx >= 0 && idx < n_q_);
      x0[idx] = std::clamp(
          predicted_state[idx],
          x0[idx] - joint.max_acceleration * approx_loop_dt * approx_loop_dt,
          x0[idx] + joint.max_acceleration * approx_loop_dt * approx_loop_dt);
    }
    for (int idx = n_q_ + joint.v_start; idx < n_q_ + joint.v_start + joint.v_size; ++idx) {
      // Ensure the joint indices are within bounds
      DRAKE_DEMAND(idx >= n_q_ && idx < n_x_);
      x0[idx] = std::clamp(predicted_state[idx],
                           x0[idx] - joint.max_acceleration * approx_loop_dt,
                           x0[idx] + joint.max_acceleration * approx_loop_dt);
    }
  }
}

void C3Controller::OutputC3Solution(
    const drake::systems::Context<double>& context,
    C3Output::C3Solution* c3_solution) const {
  double start_time = input_state_timestamp_;
  double solve_time = filtered_solve_time_;

  // Retrieve the full solution from the C3 problem
  std::vector<Eigen::VectorXd> z_sol = c3_->GetFullSolution();

  // Populate the solution output
  for (int i = 0; i < N_; i++) {
    c3_solution->time_vector_(i) = start_time + solve_time + i * dt_;
    c3_solution->x_sol_.col(i) = z_sol[i].segment(0, n_x_).cast<float>();
    c3_solution->lambda_sol_.col(i) =
        z_sol[i].segment(n_x_, n_lambda_).cast<float>();
    c3_solution->u_sol_.col(i) =
        z_sol[i].segment(n_x_ + n_lambda_, n_u_).cast<float>();
  }
}

void C3Controller::OutputC3Intermediates(
    const drake::systems::Context<double>& context,
    C3Output::C3Intermediates* c3_intermediates) const {
  double start_time = input_state_timestamp_;
  double solve_time = filtered_solve_time_;

  // Retrieve intermediate solutions from the C3 problem
  std::vector<Eigen::VectorXd> z = c3_->GetFullSolution();
  std::vector<Eigen::VectorXd> delta = c3_->GetDualDeltaSolution();
  std::vector<Eigen::VectorXd> w = c3_->GetDualWSolution();

  // Populate the intermediates output
  for (int i = 0; i < N_; i++) {
    c3_intermediates->time_vector_(i) = start_time + solve_time + i * dt_;
    c3_intermediates->z_.col(i) = z[i].cast<float>();
    c3_intermediates->delta_.col(i) = delta[i].cast<float>();
    c3_intermediates->w_.col(i) = w[i].cast<float>();
  }
}

}  // namespace systems
}  // namespace c3
