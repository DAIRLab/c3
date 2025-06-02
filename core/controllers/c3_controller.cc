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
    C3ControllerOptions c3_options)
    : plant_(plant),
      c3_options_(std::move(c3_options)),
      N_(c3_options_.N),
      publish_frequency_(c3_options.publish_frequency) {
  this->set_name("c3_controller");

  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_u_ = plant_.num_actuators();
  n_x_ = n_q_ + n_v_;
  dt_ = c3_options_.dt;
  solve_time_filter_constant_ = c3_options_.solve_time_filter_alpha;
  if (c3_options_.contact_model == "stewart_and_trinkle") {
    n_lambda_ =
        2 * c3_options_.num_contacts +
        2 * c3_options_.num_friction_directions * c3_options_.num_contacts;
  } else if (c3_options_.contact_model == "anitescu") {
    n_lambda_ =
        2 * c3_options_.num_friction_directions * c3_options_.num_contacts;
  }
  VectorXd zeros = VectorXd::Zero(n_x_ + n_lambda_ + n_u_);

  n_u_ = plant_.num_actuators();

  // Creates placeholder lcs to construct base C3 problem
  // Placeholder LCS will have correct size as it's already determined by the
  // contact model
  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, N_, dt_);
  auto x_desired_placeholder =
      std::vector<VectorXd>(N_ + 1, VectorXd::Zero(n_x_));
  if (c3_options_.projection_type == "MIQP") {
    c3_ = std::make_unique<C3MIQP>(lcs_placeholder, x_desired_placeholder,
                                   c3_options_);
  } else if (c3_options_.projection_type == "QP") {
    c3_ = std::make_unique<C3QP>(lcs_placeholder, x_desired_placeholder,
                                 c3_options_);

  } else {
    std::cerr << ("Unknown projection type") << std::endl;
    DRAKE_THROW_UNLESS(false);
  }

  c3_->SetOsqpSolverOptions(solver_options_);

  lcs_state_input_port_ =
      this->DeclareVectorInputPort("x_lcs", n_x_).get_index();
  lcs_input_port_ =
      this->DeclareAbstractInputPort("lcs", drake::Value<LCS>(lcs_placeholder))
          .get_index();

  target_input_port_ =
      this->DeclareVectorInputPort("x_lcs_des", n_x_).get_index();

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

  plan_start_time_index_ = DeclareDiscreteState(1);
  x_pred_index_ = DeclareDiscreteState(n_x_);
  filtered_solve_time_index_ = DeclareDiscreteState(1);

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
  const BasicVector<double>& x_des =
      *this->template EvalVectorInput<BasicVector>(context, target_input_port_);
  const BasicVector<double>& lcs_x =
      *this->template EvalVectorInput<BasicVector>(context,
                                                   lcs_state_input_port_);
  // const TimestampedVector<double>* lcs_x =
  //     (TimestampedVector<double>*)this->EvalVectorInput(context,
  //                                                       lcs_state_input_port_);

  auto& lcs =
      this->EvalAbstractInput(context, lcs_input_port_)->get_value<LCS>();
  drake::VectorX<double> x_lcs(lcs_x.value());  //= lcs_x->get_data();
  auto& x_pred = context.get_discrete_state(x_pred_index_).value();
  auto mutable_x_pred = discrete_state->get_mutable_value(x_pred_index_);
  auto mutable_solve_time =
      discrete_state->get_mutable_value(filtered_solve_time_index_);

  // Todo: Looks like task specific initialization, to be confirmed and removed
  // if (x_lcs.segment(n_q_, 3).norm() > 0.01 && c3_options_.use_predicted_x0 &&
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

  discrete_state->get_mutable_value(plan_start_time_index_)[0] = 0;
  // lcs_x->get_timestamp();

  std::vector<VectorXd> x_desired =
      std::vector<VectorXd>(N_ + 1, x_des.value());

  c3_->UpdateLCS(lcs);
  c3_->UpdateTarget(x_desired);
  c3_->Solve(x_lcs);

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = finish - start;
  double solve_time =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1e6;
  mutable_solve_time[0] = (1 - solve_time_filter_constant_) * solve_time +
                          solve_time_filter_constant_ * mutable_solve_time[0];

  if (publish_frequency_ > 0) {
    solve_time = 1.0 / publish_frequency_;
    mutable_solve_time[0] = solve_time;
  }

  auto z_sol = c3_->GetFullSolution();
  if (mutable_solve_time[0] < (N_ - 1) * dt_) {
    int index = mutable_solve_time[0] / dt_;
    double weight = (mutable_solve_time[0] - index * dt_) / dt_;
    mutable_x_pred = (1 - weight) * z_sol[index].segment(0, n_x_) +
                     weight * z_sol[index + 1].segment(0, n_x_);
  } else {
    mutable_x_pred = z_sol[N_ - 1].segment(0, n_x_);
  }

  return drake::systems::EventStatus::Succeeded();
}

void C3Controller::OutputC3Solution(
    const drake::systems::Context<double>& context,
    C3Output::C3Solution* c3_solution) const {
  double t = context.get_discrete_state(plan_start_time_index_)[0];
  double solve_time = context.get_discrete_state(filtered_solve_time_index_)[0];

  auto z_sol = c3_->GetFullSolution();
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
  double solve_time = context.get_discrete_state(filtered_solve_time_index_)[0];
  double t = context.get_discrete_state(plan_start_time_index_)[0] + solve_time;
  auto z = c3_->GetFullSolution();
  auto delta = c3_->GetDualDeltaSolution();
  auto w = c3_->GetDualWSolution();

  for (int i = 0; i < N_; i++) {
    c3_intermediates->time_vector_(i) = solve_time + t + i * dt_;
    c3_intermediates->z_.col(i) = z[i].cast<float>();
    c3_intermediates->delta_.col(i) = delta[i].cast<float>();
    c3_intermediates->w_.col(i) = w[i].cast<float>();
  }
}

}  // namespace systems
}  // namespace c3
