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

C3Plus::C3Plus(const LCS& LCS, const CostMatrices& costs,
               const vector<VectorXd>& xdesired, const C3Options& options)
    : C3(LCS, costs, xdesired, options, [](const class LCS& lcs) {
        return (lcs.num_states() + 2 * lcs.num_lambdas() + lcs.num_inputs());
      }) {
  eta_ = vector<drake::solvers::VectorXDecisionVariable>();
  eta_sol_ = std::make_unique<std::vector<VectorXd>>();
  for (int i = 0; i < N_; ++i) {
    eta_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    eta_.push_back(
        prog_.NewContinuousVariables(n_lambda_, "eta" + std::to_string(i)));
    z_.at(i).push_back(eta_.back());
  }
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

  // Handle complementarity constraints for each lambda-eta pair
  for (int i = 0; i < n_lambda_; ++i) {
    double w_eta =
        std::abs(U(n_x_ + n_lambda_ + n_u_ + i, n_x_ + n_lambda_ + n_u_ + i));
    double w_lambda = std::abs(U(n_x_ + i, n_x_ + i));

    double lambda_val = delta_c(n_x_ + i);
    double eta_val = delta_c(n_x_ + n_lambda_ + n_u_ + i);

    if (lambda_val <= 0) {
      delta_proj(n_x_ + i) = 0;
      delta_proj(n_x_ + n_lambda_ + n_u_ + i) = std::max(0.0, eta_val);
    } else {
      if (eta_val <= 0) {
        delta_proj(n_x_ + i) = lambda_val;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
      } else {
        // If point (lambda, eta) is above the slope sqrt(w_lambda/w_eta), set
        // lambda to 0 and keep eta Otherwise, set lambda to lambda and set eta
        // to 0
        if (eta_val * std::sqrt(w_eta) > lambda_val * std::sqrt(w_lambda)) {
          delta_proj(n_x_ + i) = 0;
          delta_proj(n_x_ + n_lambda_ + n_u_ + i) = eta_val;
        } else {
          delta_proj(n_x_ + i) = lambda_val;
          delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
        }
      }
    }
  }

  return delta_proj;
}

}  // namespace c3
