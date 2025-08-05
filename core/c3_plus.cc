#include "c3_plus.h"

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "lcs.h"

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgramResult;

C3Plus::C3Plus(const LCS& lcs, const CostMatrices& costs,
               const vector<VectorXd>& xdesired, const C3Options& options)
    : C3(lcs, costs, xdesired, options,
         lcs.num_states() + 2 * lcs.num_lambdas() + lcs.num_inputs()) {
  // Initialize eta as optimization variables
  eta_ = vector<drake::solvers::VectorXDecisionVariable>();
  eta_sol_ = std::make_unique<std::vector<VectorXd>>();
  for (int i = 0; i < N_; ++i) {
    eta_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    eta_.push_back(
        prog_.NewContinuousVariables(n_lambda_, "eta" + std::to_string(i)));
    z_.at(i).push_back(eta_.back());
  }

  // Add eta equality constraints η = E * x + F * λ + H * u + c
  MatrixXd EtaLinEq(n_lambda_, n_x_ + 2 * n_lambda_ + n_u_);
  EtaLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_lambda_, n_lambda_) =
      -1 * MatrixXd::Identity(n_lambda_, n_lambda_);
  eta_constraints_.resize(N_);
  for (int i = 0; i < N_; ++i) {
    EtaLinEq.block(0, 0, n_lambda_, n_x_) = lcs_.E().at(i);
    EtaLinEq.block(0, n_x_, n_lambda_, n_lambda_) = lcs_.F().at(i);
    EtaLinEq.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = lcs_.H().at(i);

    eta_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                EtaLinEq, -lcs_.c().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), eta_.at(i)})
            .evaluator()
            .get();
  }

  // Disable parallelization for C3+ because of the overhead cost
  use_parallelization_in_projection_ = false;
}

void C3Plus::UpdateLCS(const LCS& lcs) {
  C3::UpdateLCS(lcs);
  MatrixXd EtaLinEq(n_lambda_, n_x_ + 2 * n_lambda_ + n_u_);
  EtaLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_lambda_, n_lambda_) =
      -1 * MatrixXd::Identity(n_lambda_, n_lambda_);
  for (int i = 0; i < N_; ++i) {
    EtaLinEq.block(0, 0, n_lambda_, n_x_) = lcs_.E().at(i);
    EtaLinEq.block(0, n_x_, n_lambda_, n_lambda_) = lcs_.F().at(i);
    EtaLinEq.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = lcs_.H().at(i);
    eta_constraints_[i]->UpdateCoefficients(EtaLinEq, -lcs_.c().at(i));
  }
}

void C3Plus::WarmStartQP(const Eigen::VectorXd& x0, int admm_iteration) {
  if (admm_iteration == 0) return;  // No warm start for the first iteration
  int index = solve_time_ / lcs_.dt();
  double weight = (solve_time_ - index * lcs_.dt()) / lcs_.dt();
  for (int i = 0; i < N_ - 1; ++i) {
    prog_.SetInitialGuess(
        x_[i], (1 - weight) * warm_start_x_[admm_iteration - 1][i] +
                   weight * warm_start_x_[admm_iteration - 1][i + 1]);
    prog_.SetInitialGuess(
        lambda_[i], (1 - weight) * warm_start_lambda_[admm_iteration - 1][i] +
                        weight * warm_start_lambda_[admm_iteration - 1][i + 1]);
    prog_.SetInitialGuess(
        u_[i], (1 - weight) * warm_start_u_[admm_iteration - 1][i] +
                   weight * warm_start_u_[admm_iteration - 1][i + 1]);
    prog_.SetInitialGuess(
        eta_[i], (1 - weight) * warm_start_eta_[admm_iteration - 1][i] +
                     weight * warm_start_eta_[admm_iteration - 1][i + 1]);
  }
  prog_.SetInitialGuess(x_[N_], warm_start_x_[admm_iteration - 1][N_]);
}

void C3Plus::ProcessQPResults(const MathematicalProgramResult& result,
                              int admm_iteration, bool is_final_solve) {
  C3::ProcessQPResults(result, admm_iteration, is_final_solve);
  for (int i = 0; i < N_; i++) {
    if (is_final_solve) {
      eta_sol_->at(i) = result.GetSolution(eta_[i]);
    }
    z_sol_->at(i).segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
        result.GetSolution(eta_[i]);
  }
}

VectorXd C3Plus::SolveSingleProjection(const MatrixXd& U,
                                       const VectorXd& delta_c,
                                       const MatrixXd& E, const MatrixXd& F,
                                       const MatrixXd& H, const VectorXd& c,
                                       const int admm_iteration,
                                       const int& warm_start_index) {
  VectorXd delta_proj = delta_c;

  // Extract the weight vectors for lambda and eta from the diagonal of the cost
  // matrix U.
  VectorXd w_eta_vec = U.block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                               n_lambda_, n_lambda_)
                           .diagonal();
  VectorXd w_lambda_vec = U.block(n_x_, n_x_, n_lambda_, n_lambda_).diagonal();

  // Throw an error if any weights are negative.
  if (w_eta_vec.minCoeff() < 0 || w_lambda_vec.minCoeff() < 0) {
    throw std::runtime_error(
        "Negative weights in the cost matrix U are not allowed.");
  }

  VectorXd lambda_c = delta_c.segment(n_x_, n_lambda_);
  VectorXd eta_c = delta_c.segment(n_x_ + n_lambda_ + n_u_, n_lambda_);

  // Set the smaller of lambda and eta to zero
  Eigen::Array<bool, Eigen::Dynamic, 1> eta_larger =
      eta_c.array() * w_eta_vec.array().sqrt() >
      lambda_c.array() * w_lambda_vec.array().sqrt();

  delta_proj.segment(n_x_, n_lambda_) =
      eta_larger.select(VectorXd::Zero(n_lambda_), lambda_c);
  delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
      eta_larger.select(eta_c, VectorXd::Zero(n_lambda_));

  // Clip lambda and eta at 0
  delta_proj.segment(n_x_, n_lambda_) =
      delta_proj.segment(n_x_, n_lambda_).cwiseMax(0);
  delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
      delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_).cwiseMax(0);

  return delta_proj;
}
}  // namespace c3
