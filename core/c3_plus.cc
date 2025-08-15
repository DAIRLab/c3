#include "c3_plus.h"

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "common/logging_utils.hpp"
#include "lcs.h"

#include "drake/common/text_logging.h"

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgramResult;

C3Plus::C3Plus(const LCS& lcs, const CostMatrices& costs,
               const vector<VectorXd>& xdesired, const C3Options& options)
    : C3(lcs, costs, xdesired, options,
         lcs.num_states() + 2 * lcs.num_lambdas() + lcs.num_inputs()) {
  if (warm_start_) {
    warm_start_eta_.resize(options_.admm_iter + 1);
    for (int iter = 0; iter < options_.admm_iter + 1; ++iter) {
      warm_start_eta_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_eta_[iter][i] = VectorXd::Zero(n_lambda_);
      }
    }
  }

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
        prog_.AddLinearEqualityConstraint(EtaLinEq, -lcs_.c().at(i), z_.at(i))
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

void C3Plus::SetInitialGuessQP(const Eigen::VectorXd& x0, int admm_iteration) {
  C3::SetInitialGuessQP(x0, admm_iteration);
  if (!warm_start_ || admm_iteration == 0)
    return;  // No warm start for the first iteration
  int index = solve_time_ / lcs_.dt();
  double weight = (solve_time_ - index * lcs_.dt()) / lcs_.dt();
  for (int i = 0; i < N_ - 1; ++i) {
    prog_.SetInitialGuess(
        eta_[i], (1 - weight) * warm_start_eta_[admm_iteration - 1][i] +
                     weight * warm_start_eta_[admm_iteration - 1][i + 1]);
  }
}

void C3Plus::StoreQPResults(const MathematicalProgramResult& result,
                            int admm_iteration, bool is_final_solve) {
  C3::StoreQPResults(result, admm_iteration, is_final_solve);
  drake::log()->trace("C3Plus::StoreQPResults storing eta results.");
  for (int i = 0; i < N_; i++) {
    if (is_final_solve) {
      eta_sol_->at(i) = result.GetSolution(eta_[i]);
    }
    z_sol_->at(i).segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
        result.GetSolution(eta_[i]);
    drake::log()->trace(
        "C3Plus::StoreQPResults storing solution for time step {}:  "
        "eta = {}",
        i, EigenToString(eta_sol_->at(i).transpose()));
  }

  if (!warm_start_)
    return;  // No warm start, so no need to update warm start parameters

  drake::log()->trace("C3Plus::StoreQPResults storing warm start eta.");
  for (int i = 0; i < N_; ++i) {
    warm_start_eta_[admm_iteration][i] = result.GetSolution(eta_[i]);
  }
}

void C3Plus::AddAugmentedCost(const vector<MatrixXd>& G,
                              const vector<VectorXd>& WD,
                              const vector<VectorXd>& delta,
                              bool is_final_solve) {
  // Remove previous augmented costs
  for (auto& cost : augmented_costs_) {
    prog_.RemoveCost(cost);
  }
  augmented_costs_.clear();

  // Add augmented costs for each time step
  for (int i = 0; i < N_; i++) {
    Eigen::VectorXd GScaling = Eigen::VectorXd::Ones(n_z_);

    // Apply contact scaling to final solve if specified in options
    if (is_final_solve &&
        options_.final_augmented_cost_contact_scaling.has_value() &&
        options_.final_augmented_cost_contact_indices.has_value()) {
      for (int index : options_.final_augmented_cost_contact_indices.value()) {
        // Scale lambda or eta based on whether the corresponding delta is zero
        if (delta.at(i)[n_x_ + index] == 0)
          GScaling(n_x_ + index) *=
              options_.final_augmented_cost_contact_scaling.value();
        else
          GScaling(n_x_ + n_lambda_ + n_u_ + index) *=
              options_.final_augmented_cost_contact_scaling.value();
      }
    }

    // Apply scaling to contact forces in final solve to encourage
    // complementarity constraints. For final solve, scale G matrix by GScaling;
    // otherwise use unscaled G matrix. This helps satisfy complementarity by
    // encouraging some contact forces to match previous projection step values,
    // improving dynamic feasibility. Applied selectively to avoid
    // over-constraining the QP solver.
    const MatrixXd& G_i =
        is_final_solve ? (G.at(i).array().colwise() * GScaling.array()).matrix()
                       : G.at(i);
    const VectorXd& WD_i = is_final_solve ? delta.at(i) : WD.at(i);

    // Add quadratic cost for lambda
    augmented_costs_.push_back(prog_.AddQuadraticCost(
        2 * G_i.block(n_x_, n_x_, n_lambda_, n_lambda_),
        -2 * G_i.block(n_x_, n_x_, n_lambda_, n_lambda_) *
            WD_i.segment(n_x_, n_lambda_),
        lambda_.at(i), 1));
    // Add quadratic cost for eta
    augmented_costs_.push_back(prog_.AddQuadraticCost(
        2 * G_i.block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                      n_lambda_, n_lambda_),
        -2 *
            G_i.block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                      n_lambda_, n_lambda_) *
            WD_i.segment(n_x_ + n_lambda_ + n_u_, n_lambda_),
        eta_.at(i), 1));
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

  drake::log()->trace(
      "C3Plus::SolveSingleProjection ADMM iteration {}: pre-projection "
      "lambda = {}, eta = {}",
      admm_iteration, EigenToString(lambda_c.transpose()),
      EigenToString(eta_c.transpose()));

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
