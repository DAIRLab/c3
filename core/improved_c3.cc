#include "improved_c3.h"

#include <cfenv> // for fesetround
#include <chrono>
#include <iostream>

#include <Eigen/Core>
#include <omp.h>

#include "lcs.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/moby_lcp_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;
using drake::solvers::SolverOptions;

using drake::solvers::OsqpSolver;
using drake::solvers::OsqpSolverDetails;
using drake::solvers::Solve;

ImprovedC3::CostMatrices::CostMatrices(const std::vector<Eigen::MatrixXd> &Q,
                                       const std::vector<Eigen::MatrixXd> &R,
                                       const std::vector<Eigen::MatrixXd> &G,
                                       const std::vector<Eigen::MatrixXd> &U) {
  this->Q = Q;
  this->R = R;
  this->G = G;
  this->U = U;
}

ImprovedC3::ImprovedC3(const LCS &lcs, const ImprovedC3::CostMatrices &costs,
                       const vector<VectorXd> &x_desired,
                       const C3Options &options)
    : warm_start_(options.warm_start), N_(lcs.N()), n_x_(lcs.num_states()),
      n_lambda_(lcs.num_lambdas()), n_u_(lcs.num_inputs()), lcs_(lcs),
      cost_matrices_(costs), x_desired_(x_desired), options_(options),
      h_is_zero_(lcs.H()[0].isZero(0)), prog_(MathematicalProgram()),
      osqp_(OsqpSolver()) {
  if (warm_start_) {
    warm_start_delta_.resize(options_.admm_iter + 1);
    warm_start_x_.resize(options_.admm_iter + 1);
    warm_start_lambda_.resize(options_.admm_iter + 1);
    warm_start_u_.resize(options_.admm_iter + 1);
    for (int iter = 0; iter < options_.admm_iter + 1; ++iter) {
      warm_start_delta_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_delta_[iter][i] =
            VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_);
      }
      warm_start_x_[iter].resize(N_ + 1);
      for (int i = 0; i < N_ + 1; ++i) {
        warm_start_x_[iter][i] = VectorXd::Zero(n_x_);
      }
      warm_start_lambda_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_lambda_[iter][i] = VectorXd::Zero(n_lambda_);
      }
      warm_start_u_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_u_[iter][i] = VectorXd::Zero(n_u_);
      }
    }
  }

  ScaleLCS();
  x_ = vector<drake::solvers::VectorXDecisionVariable>();
  u_ = vector<drake::solvers::VectorXDecisionVariable>();
  lambda_ = vector<drake::solvers::VectorXDecisionVariable>();
  gamma_ = vector<drake::solvers::VectorXDecisionVariable>();

  z_sol_ = std::make_unique<std::vector<VectorXd>>();
  x_sol_ = std::make_unique<std::vector<VectorXd>>();
  lambda_sol_ = std::make_unique<std::vector<VectorXd>>();
  u_sol_ = std::make_unique<std::vector<VectorXd>>();
  gamma_sol_ = std::make_unique<std::vector<VectorXd>>();
  w_sol_ = std::make_unique<std::vector<VectorXd>>();
  delta_sol_ = std::make_unique<std::vector<VectorXd>>();
  for (int i = 0; i < N_; ++i) {
    z_sol_->push_back(Eigen::VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
    x_sol_->push_back(Eigen::VectorXd::Zero(n_x_));
    lambda_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    u_sol_->push_back(Eigen::VectorXd::Zero(n_u_));
    gamma_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    w_sol_->push_back(Eigen::VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
    delta_sol_->push_back(Eigen::VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
  }

  for (int i = 0; i < N_ + 1; ++i) {
    x_.push_back(prog_.NewContinuousVariables(n_x_, "x" + std::to_string(i)));
    if (i < N_) {
      u_.push_back(prog_.NewContinuousVariables(n_u_, "k" + std::to_string(i)));
      lambda_.push_back(prog_.NewContinuousVariables(
          n_lambda_, "lambda" + std::to_string(i)));
      gamma_.push_back(
          prog_.NewContinuousVariables(n_lambda_, "gamma" + std::to_string(i)));
    }
  }

  // initialize the constraint bindings
  initial_state_constraint_ = nullptr;
  initial_force_constraint_ = nullptr;
  dynamics_constraints_.resize(N_);
  sdf_constraints_.resize(N_);
  target_cost_.resize(N_ + 1);

  MatrixXd DynamicsLinEq(n_x_, 2 * n_x_ + n_lambda_ + n_u_);
  DynamicsLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_x_, n_x_) =
      -1 * MatrixXd::Identity(n_x_, n_x_);
  MatrixXd SDFLinEq(n_lambda_, n_x_ + 2 * n_lambda_ + n_u_);
  SDFLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_lambda_, n_lambda_) =
      -1 * MatrixXd::Identity(n_lambda_, n_lambda_);
  for (int i = 0; i < N_; ++i) {
    DynamicsLinEq.block(0, 0, n_x_, n_x_) = lcs.A().at(i);
    DynamicsLinEq.block(0, n_x_, n_x_, n_lambda_) = lcs.D().at(i);
    DynamicsLinEq.block(0, n_x_ + n_lambda_, n_x_, n_u_) = lcs.B().at(i);

    dynamics_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                DynamicsLinEq, -lcs_.d().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), x_.at(i + 1)})
            .evaluator()
            .get();

    SDFLinEq.block(0, 0, n_lambda_, n_x_) = lcs.E().at(i);
    SDFLinEq.block(0, n_x_, n_lambda_, n_lambda_) = lcs.F().at(i);
    SDFLinEq.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = lcs.H().at(i);

    sdf_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                SDFLinEq, -lcs_.c().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), gamma_.at(i)})
            .evaluator()
            .get();
  }
  input_costs_.resize(N_);
  for (int i = 0; i < N_ + 1; ++i) {
    target_cost_[i] =
        prog_
            .AddQuadraticCost(2 * cost_matrices_.Q.at(i),
                              -2 * cost_matrices_.Q.at(i) * x_desired_.at(i),
                              x_.at(i), 1)
            .evaluator()
            .get();
    if (i < N_) {
      input_costs_[i] = prog_
                            .AddQuadraticCost(2 * cost_matrices_.R.at(i),
                                              VectorXd::Zero(n_u_), u_.at(i), 1)
                            .evaluator();
    }
  }
}

void ImprovedC3::ScaleLCS() {
  DRAKE_DEMAND(lcs_.D()[0].norm() > 0);
  double Dn = lcs_.D()[0].norm();
  double An = lcs_.A()[0].norm();
  AnDn_ = An / Dn;
  lcs_.ScaleComplementarityDynamics(AnDn_);
}

void ImprovedC3::UpdateLCS(const LCS &lcs) {
  DRAKE_DEMAND(lcs_.HasSameDimensionsAs(lcs));

  lcs_ = lcs;
  h_is_zero_ = lcs_.H()[0].isZero(0);
  ScaleLCS();

  MatrixXd LinEq = MatrixXd::Zero(n_x_, 2 * n_x_ + n_lambda_ + n_u_);
  LinEq.block(0, n_x_ + n_u_ + n_lambda_, n_x_, n_x_) =
      -1 * MatrixXd::Identity(n_x_, n_x_);
  for (int i = 0; i < N_; ++i) {
    LinEq.block(0, 0, n_x_, n_x_) = lcs_.A().at(i);
    LinEq.block(0, n_x_, n_x_, n_lambda_) = lcs_.D().at(i);
    LinEq.block(0, n_x_ + n_lambda_, n_x_, n_u_) = lcs_.B().at(i);

    dynamics_constraints_[i]->UpdateCoefficients(LinEq, -lcs.d().at(i));
  }
}

const std::vector<drake::solvers::LinearEqualityConstraint *> &
ImprovedC3::GetDynamicConstraints() {
  return dynamics_constraints_;
}

void ImprovedC3::UpdateTarget(const std::vector<Eigen::VectorXd> &x_des) {
  x_desired_ = x_des;
  for (int i = 0; i < N_ + 1; ++i) {
    target_cost_[i]->UpdateCoefficients(2 * cost_matrices_.Q.at(i),
                                        -2 * cost_matrices_.Q.at(i) *
                                            x_desired_.at(i));
  }
}

const std::vector<drake::solvers::QuadraticCost *> &
ImprovedC3::GetTargetCost() {
  return target_cost_;
}

void ImprovedC3::Solve(const VectorXd &x0) {
  auto start = std::chrono::high_resolution_clock::now();
  if (initial_state_constraint_) {
    initial_state_constraint_->UpdateCoefficients(
        MatrixXd::Identity(n_x_, n_x_), x0);
  } else {
    initial_state_constraint_ =
        prog_
            .AddLinearEqualityConstraint(MatrixXd::Identity(n_x_, n_x_), x0,
                                         x_[0])
            .evaluator();
  }
  VectorXd delta_init = VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_);
  if (options_.delta_option == 1) {
    delta_init.head(n_x_) = x0;
  }
  std::vector<VectorXd> delta(N_, delta_init);
  std::vector<VectorXd> w(N_, VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
  vector<MatrixXd> G = cost_matrices_.G;

  for (int i = 0; i < N_; ++i) {
    input_costs_[i]->UpdateCoefficients(2 * cost_matrices_.R.at(i),
                                        -2 * cost_matrices_.R.at(i) *
                                            u_sol_->at(i));
  }

  for (int iter = 0; iter < options_.admm_iter; iter++) {
    ADMMStep(x0, &delta, &w, &G, iter);
  }

  vector<VectorXd> WD(N_, VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta.at(i) - w.at(i);
  }

  vector<VectorXd> zfin = SolveQP(x0, G, WD, options_.admm_iter, true);

  *w_sol_ = w;
  *delta_sol_ = delta;

  if (!options_.end_on_qp_step) {
    *z_sol_ = delta;
    z_sol_->at(0).segment(0, n_x_) = x0;
    x_sol_->at(0) = x0;
    for (int i = 1; i < N_; ++i) {
      z_sol_->at(i).segment(0, n_x_) =
          lcs_.A().at(i - 1) * x_sol_->at(i - 1) +
          lcs_.B().at(i - 1) * u_sol_->at(i - 1) +
          lcs_.D().at(i - 1) * lambda_sol_->at(i - 1) + lcs_.d().at(i - 1);
    }
  }

  // Undoing to scaling to put variables back into correct units
  // This only scales lambda
  for (int i = 0; i < N_; ++i) {
    lambda_sol_->at(i) *= AnDn_;
    z_sol_->at(i).segment(n_x_, n_lambda_) *= AnDn_;
  }

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = finish - start;
  solve_time_ =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1e6;
}

void ImprovedC3::ADMMStep(const VectorXd &x0, vector<VectorXd> *delta,
                          vector<VectorXd> *w, vector<MatrixXd> *G,
                          int admm_iteration) {
  vector<VectorXd> WD(N_, VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));

  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta->at(i) - w->at(i);
  }

  vector<VectorXd> z = SolveQP(x0, *G, WD, admm_iteration, true);

  vector<VectorXd> ZW(N_, VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
  for (int i = 0; i < N_; ++i) {
    ZW[i] = w->at(i) + z[i];
  }

  if (cost_matrices_.U[0].isZero(0)) {
    *delta = SolveProjection(*G, ZW, admm_iteration);

  } else {
    *delta = SolveProjection(cost_matrices_.U, ZW, admm_iteration);
  }

  for (int i = 0; i < N_; ++i) {
    w->at(i) = w->at(i) + z[i] - delta->at(i);
    w->at(i) = w->at(i) / options_.rho_scale;
    G->at(i) = G->at(i) * options_.rho_scale;
  }
}

vector<VectorXd> ImprovedC3::SolveQP(const VectorXd &x0,
                                     const vector<MatrixXd> &G,
                                     const vector<VectorXd> &WD,
                                     int admm_iteration, bool is_final_solve) {
  if (h_is_zero_ == 1) { // No dependence on u, so just simulate passive system
    drake::solvers::MobyLCPSolver<double> LCPSolver;
    VectorXd lambda0;
    LCPSolver.SolveLcpLemke(lcs_.F()[0], lcs_.E()[0] * x0 + lcs_.c()[0],
                            &lambda0);
    if (initial_force_constraint_) {
      initial_force_constraint_->UpdateCoefficients(
          MatrixXd::Identity(n_lambda_, n_lambda_), lambda0);
    } else {
      initial_force_constraint_ =
          prog_
              .AddLinearEqualityConstraint(
                  MatrixXd::Identity(n_lambda_, n_lambda_), lambda0, lambda_[0])
              .evaluator();
    }
  }

  for (auto &cost : costs_) {
    prog_.RemoveCost(cost);
  }
  costs_.clear();

  for (int i = 0; i < N_ + 1; ++i) {
    if (i < N_) {
      costs_.push_back(prog_.AddQuadraticCost(
          2 * G.at(i).block(n_x_, n_x_, n_lambda_, n_lambda_),
          -2 * G.at(i).block(n_x_, n_x_, n_lambda_, n_lambda_) *
              WD.at(i).segment(n_x_, n_lambda_),
          lambda_.at(i), 1));
      costs_.push_back(prog_.AddQuadraticCost(
          2 * G.at(i).block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                            n_lambda_, n_lambda_),
          -2 *
              G.at(i).block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                            n_lambda_, n_lambda_) *
              WD.at(i).segment(n_x_ + n_lambda_ + n_u_, n_lambda_),
          gamma_.at(i), 1));
    }
  }

  if (warm_start_) {
    int index = solve_time_ / lcs_.dt();
    double weight = (solve_time_ - index * lcs_.dt()) / lcs_.dt();
    for (int i = 0; i < N_ - 1; ++i) {
      prog_.SetInitialGuess(x_[i],
                            (1 - weight) * warm_start_x_[admm_iteration][i] +
                                weight * warm_start_x_[admm_iteration][i + 1]);
      prog_.SetInitialGuess(
          lambda_[i], (1 - weight) * warm_start_lambda_[admm_iteration][i] +
                          weight * warm_start_lambda_[admm_iteration][i + 1]);
      prog_.SetInitialGuess(u_[i],
                            (1 - weight) * warm_start_u_[admm_iteration][i] +
                                weight * warm_start_u_[admm_iteration][i + 1]);
    }
    prog_.SetInitialGuess(x_[0], x0);
    prog_.SetInitialGuess(x_[N_], warm_start_x_[admm_iteration][N_]);
  }

  MathematicalProgramResult result = osqp_.Solve(prog_);

  if (result.is_success()) {
    for (int i = 0; i < N_; ++i) {
      if (is_final_solve) {
        x_sol_->at(i) = result.GetSolution(x_[i]);
        lambda_sol_->at(i) = result.GetSolution(lambda_[i]);
        u_sol_->at(i) = result.GetSolution(u_[i]);
        gamma_sol_->at(i) = result.GetSolution(gamma_[i]);
      }
      z_sol_->at(i).segment(0, n_x_) = result.GetSolution(x_[i]);
      z_sol_->at(i).segment(n_x_, n_lambda_) = result.GetSolution(lambda_[i]);
      z_sol_->at(i).segment(n_x_ + n_lambda_, n_u_) = result.GetSolution(u_[i]);
      z_sol_->at(i).segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
          result.GetSolution(gamma_[i]);

      if (warm_start_) {
        // update warm start parameters
        warm_start_x_[admm_iteration][i] = result.GetSolution(x_[i]);
        warm_start_lambda_[admm_iteration][i] = result.GetSolution(lambda_[i]);
        warm_start_u_[admm_iteration][i] = result.GetSolution(u_[i]);
      }
    }
    if (warm_start_) {
      warm_start_x_[admm_iteration][N_] = result.GetSolution(x_[N_]);
    }
  } else {
    std::cout << "QP failed to solve" << std::endl;
  }

  return *z_sol_;
}

vector<VectorXd> ImprovedC3::SolveProjection(const vector<MatrixXd> &U,
                                             vector<VectorXd> &WZ,
                                             int admm_iteration) {
  vector<VectorXd> deltaProj(N_, VectorXd::Zero(n_x_ + 2 * n_lambda_ + n_u_));
  int i;

  if (options_.num_threads > 0) {
    omp_set_dynamic(0); // Explicitly disable dynamic teams
    omp_set_num_threads(options_.num_threads); // Set number of threads
    omp_set_nested(0);
    omp_set_schedule(omp_sched_static, 0);
  }

#pragma omp parallel for num_threads(options_.num_threads)
  for (i = 0; i < N_; ++i) {
    if (warm_start_) {
      if (i == N_ - 1) {
        deltaProj[i] =
            SolveSingleProjection(U[i], WZ[i], lcs_.E()[i], lcs_.F()[i],
                                  lcs_.H()[i], lcs_.c()[i], admm_iteration, -1);
      } else {
        deltaProj[i] = SolveSingleProjection(
            U[i], WZ[i], lcs_.E()[i], lcs_.F()[i], lcs_.H()[i], lcs_.c()[i],
            admm_iteration, i + 1);
      }
    } else {
      deltaProj[i] =
          SolveSingleProjection(U[i], WZ[i], lcs_.E()[i], lcs_.F()[i],
                                lcs_.H()[i], lcs_.c()[i], admm_iteration, -1);
    }
  }

  return deltaProj;
}

VectorXd ImprovedC3::SolveSingleProjection(const MatrixXd &U,
                                           const VectorXd &delta_c,
                                           const MatrixXd &E, const MatrixXd &F,
                                           const MatrixXd &H, const VectorXd &c,
                                           const int admm_iteration,
                                           const int &warm_start_index) {
  // Initialize result vector with input values
  VectorXd delta_proj = delta_c;

  // Set epsilon for floating point comparisons
  const double EPSILON = 1e-8;

  // Set rounding mode to nearest
  std::fesetround(FE_TONEAREST);

  // Handle complementarity constraints for each lambda-gamma pair
  for (int i = 0; i < n_lambda_; ++i) {
    // Calculate ratio of U matrix elements for scaling with more stable
    // computation
    double u_ratio = 0.0;
    double u1 =
        std::abs(U(n_x_ + n_lambda_ + n_u_ + i, n_x_ + n_lambda_ + n_u_ + i));
    double u2 = std::abs(U(n_x_ + i, n_x_ + i));

    if (u2 < EPSILON) {
      throw std::runtime_error("Numerical instability detected: u2 is very "
                               "close to zero in SolveSingleProjection");
    }
    u_ratio = std::sqrt(u1 / u2);

    // Get current lambda and gamma values
    double lambda_val = delta_c(n_x_ + i);
    double gamma_val = delta_c(n_x_ + n_lambda_ + n_u_ + i);

    // Case 1: lambda < -EPSILON
    // In this case, we always set lambda to 0 and keep gamma if it's positive
    if (lambda_val < -EPSILON) {
      delta_proj(n_x_ + i) = 0;
      delta_proj(n_x_ + n_lambda_ + n_u_ + i) = std::max(EPSILON, gamma_val);
    }
    // Case 2: -EPSILON ≤ lambda ≤ EPSILON (lambda is very close to zero)
    else if (std::abs(lambda_val) <= EPSILON) {
      if (gamma_val < -EPSILON) {
        // If gamma is negative, set lambda to EPSILON and gamma to 0
        delta_proj(n_x_ + i) = EPSILON;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
      } else if (std::abs(gamma_val) <= EPSILON) {
        // If both are close to zero, set both to 0
        delta_proj(n_x_ + i) = 0;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
      } else {
        // If gamma is positive, set lambda to 0 and keep gamma
        delta_proj(n_x_ + i) = 0;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = gamma_val;
      }
    }
    // Case 3: lambda > EPSILON
    else {
      if (gamma_val < -EPSILON) {
        // If gamma is negative, keep lambda and set gamma to 0
        delta_proj(n_x_ + i) = lambda_val;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
      } else if (std::abs(gamma_val) <= EPSILON) {
        // If gamma is close to zero, keep lambda and set gamma to 0
        delta_proj(n_x_ + i) = lambda_val;
        delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
      } else {
        // If gamma is positive, compare with u_ratio * lambda
        if (gamma_val > u_ratio * lambda_val + EPSILON) {
          // If gamma is significantly larger, keep lambda and set gamma to 0
          delta_proj(n_x_ + i) = lambda_val;
          delta_proj(n_x_ + n_lambda_ + n_u_ + i) = 0;
        } else {
          // Otherwise, set lambda to 0 and keep gamma
          delta_proj(n_x_ + i) = 0;
          delta_proj(n_x_ + n_lambda_ + n_u_ + i) = gamma_val;
        }
      }
    }
  }

  return delta_proj;
}

void ImprovedC3::AddLinearConstraint(const Eigen::MatrixXd &A,
                                     const VectorXd &lower_bound,
                                     const VectorXd &upper_bound,
                                     int constraint) {
  if (constraint == 1) {
    for (int i = 1; i < N_; ++i) {
      user_constraints_.push_back(
          prog_.AddLinearConstraint(A, lower_bound, upper_bound, x_.at(i)));
    }
  }

  if (constraint == 2) {
    for (int i = 0; i < N_; ++i) {
      user_constraints_.push_back(
          prog_.AddLinearConstraint(A, lower_bound, upper_bound, u_.at(i)));
    }
  }

  if (constraint == 3) {
    for (int i = 0; i < N_; ++i) {
      user_constraints_.push_back(prog_.AddLinearConstraint(
          A, lower_bound, upper_bound, lambda_.at(i)));
    }
  }
}

void ImprovedC3::AddLinearConstraint(const Eigen::RowVectorXd &A,
                                     double lower_bound, double upper_bound,
                                     int constraint) {
  Eigen::VectorXd lb(1);
  lb << lower_bound;
  Eigen::VectorXd ub(1);
  ub << upper_bound;
  AddLinearConstraint(A, lb, ub, constraint);
}

void ImprovedC3::RemoveConstraints() {
  for (auto &userconstraint : user_constraints_) {
    prog_.RemoveConstraint(userconstraint);
  }
  user_constraints_.clear();
}

const std::vector<LinearConstraintBinding> &ImprovedC3::GetLinearConstraints() {
  return user_constraints_;
}

} // namespace c3