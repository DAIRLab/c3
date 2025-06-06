#include "c3_controller.h"

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

  // Determine the size of lambda based on the contact model
  if (controller_options_.contact_model == "stewart_and_trinkle") {
    n_lambda_ =
        2 * controller_options_.num_contacts +
        2 * controller_options_.num_friction_directions * controller_options_.num_contacts;
  } else if (controller_options_.contact_model == "anitescu") {
    n_lambda_ =
        2 * controller_options_.num_friction_directions * controller_options_.num_contacts;
  }

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
    std::cerr << ("Unknown projection type") << std::endl;
    DRAKE_THROW_UNLESS(false);
  }

  // Set solver options
  c3_->SetOsqpSolverOptions(solver_options_);

  // Declare input ports
  lcs_state_input_port_ =
      this->DeclareVectorInputPort("x_lcs", TimestampedVector<double>(n_x_))
          .get_index();
  lcs_input_port_ =
      this->DeclareAbstractInputPort("lcs", drake::Value<LCS>(lcs_placeholder))
          .get_index();
  target_input_port_ =
      this->DeclareVectorInputPort("x_lcs_des", n_x_).get_index();

  // Declare output ports
  c3_solution_port_ =
      this->DeclareAbstractOutputPort(
              "c3_solution", C3Output::C3Solution(n_x_, n_lambda_, n_u_, N_),
              &C3Controller::OutputC3Solution)
          .get_index();
  c3_intermediates_port_ =
      this->DeclareAbstractOutputPort(
              "c3_intermediates",
              C3Output::C3Intermediates(n_x_, n_lambda_, n_u_, N_),
              &C3Controller::OutputC3Intermediates)
          .get_index();

  // Declare discrete states
  plan_start_time_index_ = DeclareDiscreteState(1);
  // x_pred_index_ = DeclareDiscreteState(n_x_);
  filtered_solve_time_index_ = DeclareDiscreteState(1);

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
  auto start = std::chrono::high_resolution_clock::now();

  // Retrieve inputs
  const BasicVector<double>& x_des =
      *this->template EvalVectorInput<BasicVector>(context, target_input_port_);
  const TimestampedVector<double>* lcs_x =
      (TimestampedVector<double>*)this->EvalVectorInput(context,
                                                        lcs_state_input_port_);
  auto& lcs =
      this->EvalAbstractInput(context, lcs_input_port_)->get_value<LCS>();
  drake::VectorX<double> x_lcs = lcs_x->get_data();

  // TODO: Looks like task specific initialization, to be confirmed and removed
  // auto& x_pred = context.get_discrete_state(x_pred_index_).value();
  // auto mutable_x_pred = discrete_state->get_mutable_value(x_pred_index_);

  // TODO: Looks like task specific initialization, to be confirmed and removed
  // if (x_lcs.segment(n_q_, 3).norm() > 0.01 && controller_options_.use_predicted_x0 &&
  //     !x_pred.isZero()) {
  //   x_lcs[0] = std::clamp(x_pred[0], x_lcs[0] - 10 * dt_ * dt_,
  //                         x_lcs[0] + 10 * dt_ * dt_);
  //   x_lcs[1] = std::clamp(x_pred[1], x_lcs[1] - 10 * dt_ * dt_,
  //                         x_lcs[1] + 10 * dt_ * dt_);
  //   x_lcs[2] = std::clamp(x_pred[2], x_lcs[2] - 10 * dt_ * dt_,
  //                         x_lcs[2] + 10 * dt_ * dt_);
  //   x_lcs[n_q_ + 0] = std::clamp(x_pred[n_q_ + 0], x_lcs[n_q_ + 0] - 10 *
  //   dt_,
  //                                x_lcs[n_q_ + 0] + 10 * dt_);
  //   x_lcs[n_q_ + 1] = std::clamp(x_pred[n_q_ + 1], x_lcs[n_q_ + 1] - 10 *
  //   dt_,
  //                                x_lcs[n_q_ + 1] + 10 * dt_);
  //   x_lcs[n_q_ + 2] = std::clamp(x_pred[n_q_ + 2], x_lcs[n_q_ + 2] - 10 *
  //   dt_,
  //                                x_lcs[n_q_ + 2] + 10 * dt_);
  // }

  discrete_state->get_mutable_value(plan_start_time_index_)[0] =
      lcs_x->get_timestamp();

  std::vector<VectorXd> x_desired =
      std::vector<VectorXd>(N_ + 1, x_des.value());

  // Update LCS and target in the C3 problem
  c3_->UpdateLCS(lcs);
  c3_->UpdateTarget(x_desired);
  c3_->Solve(x_lcs);

  // Measure solve time
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = finish - start;
  double solve_time =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1e6;

  // Update the filtered solve time using a moving average filter
  auto mutable_solve_time =
      discrete_state->get_mutable_value(filtered_solve_time_index_);
  mutable_solve_time[0] = (1 - solve_time_filter_constant_) * solve_time +
                          solve_time_filter_constant_ * mutable_solve_time[0];

  // Override solve time if a publish frequency is specified
  if (publish_frequency_ > 0) {
    solve_time = 1.0 / publish_frequency_;
    mutable_solve_time[0] = solve_time;
  }

  // TODO: Looks like task specific initialization, to be confirmed and removed
  // std::vector<Eigen::VectorXd> z_sol = c3_->GetFullSolution();
  // if (mutable_solve_time[0] < (N_ - 1) * dt_) {
  //   int index = mutable_solve_time[0] / dt_;
  //   double weight = (mutable_solve_time[0] - index * dt_) / dt_;
  //   mutable_x_pred = (1 - weight) * z_sol[index].segment(0, n_x_) +
  //                    weight * z_sol[index + 1].segment(0, n_x_);
  // } else {
  //   mutable_x_pred = z_sol[N_ - 1].segment(0, n_x_);
  // }

  return drake::systems::EventStatus::Succeeded();
}

void C3Controller::OutputC3Solution(
    const drake::systems::Context<double>& context,
    C3Output::C3Solution* c3_solution) const {
  // Retrieve discrete state values
  double t = context.get_discrete_state(plan_start_time_index_)[0];
  double solve_time = context.get_discrete_state(filtered_solve_time_index_)[0];

  // Retrieve the full solution from the C3 problem
  std::vector<Eigen::VectorXd> z_sol = c3_->GetFullSolution();

  // Populate the solution output
  for (int i = 0; i < N_; i++) {
    c3_solution->time_vector_(i) = solve_time + t + i * dt_;
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
  // Retrieve discrete state values
  double solve_time = context.get_discrete_state(filtered_solve_time_index_)[0];
  double t = context.get_discrete_state(plan_start_time_index_)[0];

  // Retrieve intermediate solutions from the C3 problem
  std::vector<Eigen::VectorXd> z = c3_->GetFullSolution();
  std::vector<Eigen::VectorXd> delta = c3_->GetDualDeltaSolution();
  std::vector<Eigen::VectorXd> w = c3_->GetDualWSolution();

  // Populate the intermediates output
  for (int i = 0; i < N_; i++) {
    c3_intermediates->time_vector_(i) = solve_time + t + i * dt_;
    c3_intermediates->z_.col(i) = z[i].cast<float>();
    c3_intermediates->delta_.col(i) = delta[i].cast<float>();
    c3_intermediates->w_.col(i) = w[i].cast<float>();
  }
}

}  // namespace systems
}  // namespace c3
