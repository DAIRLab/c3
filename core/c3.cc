#include "c3.h"

#include <chrono>
#include <iostream>

#include <Eigen/Core>
#include <drake/common/find_runfiles.h>
#include <omp.h>

#include "lcs.h"
#include "solver_options_io.h"
#include "configs/solve_options_default.hpp"

#include "drake/common/text_logging.h"
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

C3::CostMatrices::CostMatrices(const std::vector<Eigen::MatrixXd>& Q,
                               const std::vector<Eigen::MatrixXd>& R,
                               const std::vector<Eigen::MatrixXd>& G,
                               const std::vector<Eigen::MatrixXd>& U) {
  this->Q = Q;
  this->R = R;
  this->G = G;
  this->U = U;
}

C3::C3(const LCS& lcs, const CostMatrices& costs,
       const vector<VectorXd>& x_desired, const C3Options& options,
       const int z_size)
    : warm_start_(options.warm_start),
      N_(lcs.N()),
      n_x_(lcs.num_states()),
      n_lambda_(lcs.num_lambdas()),
      n_u_(lcs.num_inputs()),
      n_z_(z_size),
      lcs_(lcs),
      cost_matrices_(costs),
      x_desired_(x_desired),
      options_(options),
      h_is_zero_(lcs.H()[0].isZero(0)),
      prog_(MathematicalProgram()),
      osqp_(OsqpSolver()) {
  if (warm_start_) {
    warm_start_x_.resize(options_.admm_iter + 1);
    warm_start_lambda_.resize(options_.admm_iter + 1);
    warm_start_u_.resize(options_.admm_iter + 1);
    for (int iter = 0; iter < options_.admm_iter + 1; ++iter) {
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

  z_sol_ = std::make_unique<std::vector<VectorXd>>();
  x_sol_ = std::make_unique<std::vector<VectorXd>>();
  lambda_sol_ = std::make_unique<std::vector<VectorXd>>();
  u_sol_ = std::make_unique<std::vector<VectorXd>>();
  w_sol_ = std::make_unique<std::vector<VectorXd>>();
  delta_sol_ = std::make_unique<std::vector<VectorXd>>();
  for (int i = 0; i < N_; ++i) {
    x_sol_->push_back(Eigen::VectorXd::Zero(n_x_));
    lambda_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    u_sol_->push_back(Eigen::VectorXd::Zero(n_u_));
    z_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
    w_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
    delta_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
  }

  z_.resize(N_);
  for (int i = 0; i < N_ + 1; ++i) {
    x_.push_back(prog_.NewContinuousVariables(n_x_, "x" + std::to_string(i)));
    if (i < N_) {
      lambda_.push_back(prog_.NewContinuousVariables(
          n_lambda_, "lambda" + std::to_string(i)));
      u_.push_back(prog_.NewContinuousVariables(n_u_, "k" + std::to_string(i)));
      z_.at(i).push_back(x_.back());
      z_.at(i).push_back(lambda_.back());
      z_.at(i).push_back(u_.back());
    }
  }

  // initialize the constraint bindings
  initial_state_constraint_ = nullptr;
  initial_force_constraint_ = nullptr;

  // Add dynamics constraints
  dynamics_constraints_.resize(N_);
  MatrixXd LinEq(n_x_, 2 * n_x_ + n_lambda_ + n_u_);
  LinEq.block(0, n_x_ + n_lambda_ + n_u_, n_x_, n_x_) =
      -1 * MatrixXd::Identity(n_x_, n_x_);
  for (int i = 0; i < N_; ++i) {
    LinEq.block(0, 0, n_x_, n_x_) = lcs_.A().at(i);
    LinEq.block(0, n_x_, n_x_, n_lambda_) = lcs_.D().at(i);
    LinEq.block(0, n_x_ + n_lambda_, n_x_, n_u_) = lcs_.B().at(i);

    dynamics_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                LinEq, -lcs_.d().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), x_.at(i + 1)})
            .evaluator()
            .get();
  }

  // Setup QP costs
  target_costs_.resize(N_ + 1);
  input_costs_.resize(N_);
  augmented_costs_.clear();
  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i] =
        prog_
            .AddQuadraticCost(2 * cost_matrices_.Q.at(i),
                              -2 * cost_matrices_.Q.at(i) * x_desired_.at(i),
                              x_.at(i), 1)
            .evaluator()
            .get();
    // Skip input cost at the (N + 1)th time step
    if (i == N_) break;
    input_costs_[i] = prog_
                          .AddQuadraticCost(2 * cost_matrices_.R.at(i),
                                            VectorXd::Zero(n_u_), u_.at(i), 1)
                          .evaluator();
  }

  // Set default solver options
  SetDefaultSolverOptions();
}

void C3::SetDefaultSolverOptions() {
  // Set default solver options
  auto main_runfile =
      drake::FindRunfile("_main/core/configs/solver_options_default.yaml");
  auto external_runfile =
      drake::FindRunfile("c3/core/configs/solver_options_default.yaml");
  if (main_runfile.abspath.empty() && external_runfile.abspath.empty()) {
    throw std::runtime_error(fmt::format(
        "Could not find the default solver options YAML file. {}, {}",
        main_runfile.error, external_runfile.error));
  }
  drake::solvers::SolverOptions solver_options =
      drake::yaml::LoadYamlFile<c3::SolverOptionsFromYaml>(
          main_runfile.abspath.empty() ? external_runfile.abspath
                                       : main_runfile.abspath)
          .GetAsSolverOptions(drake::solvers::OsqpSolver::id());
  SetSolverOptions(solver_options);
}

C3::C3(const LCS& lcs, const CostMatrices& costs,
       const vector<VectorXd>& x_desired, const C3Options& options)
    : C3(lcs, costs, x_desired, options,
         lcs.num_states() + lcs.num_lambdas() + lcs.num_inputs()) {}

C3::CostMatrices C3::CreateCostMatricesFromC3Options(const C3Options& options,
                                                     int N) {
  std::vector<Eigen::MatrixXd> Q;  // State cost matrices.
  std::vector<Eigen::MatrixXd> R;  // Input cost matrices.

  std::vector<MatrixXd> G(N,
                          options.G);  // State-input cross-term matrices.
  std::vector<MatrixXd> U(N,
                          options.U);  // Constraint matrices.

  double discount_factor = 1.0;
  for (int i = 0; i < N; ++i) {
    Q.push_back(discount_factor * options.Q);
    R.push_back(discount_factor * options.R);
    discount_factor *= options.gamma;
  }
  Q.push_back(discount_factor * options.Q);

  return CostMatrices(Q, R, G, U);  // Initialize the cost matrices.
}

void C3::ScaleLCS() {
  if (!options_.scale_lcs) {
    // If the LCS is a placeholder or scaling is disabled, we do not scale
    // the complementarity dynamics.
    AnDn_ = 1.0;
    return;
  }

  AnDn_ = lcs_.ScaleComplementarityDynamics();
}

void C3::UpdateLCS(const LCS& lcs) {
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

    dynamics_constraints_[i]->UpdateCoefficients(LinEq, -lcs_.d().at(i));
  }
}

const std::vector<drake::solvers::LinearEqualityConstraint*>&
C3::GetDynamicConstraints() {
  return dynamics_constraints_;
}

void C3::UpdateTarget(const std::vector<Eigen::VectorXd>& x_des) {
  x_desired_ = x_des;
  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i]->UpdateCoefficients(
        2 * cost_matrices_.Q.at(i),
        -2 * cost_matrices_.Q.at(i) * x_desired_.at(i));
  }
}

void C3::UpdateCostMatrices(const CostMatrices& costs) {
  DRAKE_DEMAND(cost_matrices_.HasSameDimensionsAs(costs));
  cost_matrices_ = costs;

  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i]->UpdateCoefficients(
        2 * cost_matrices_.Q.at(i),
        -2 * cost_matrices_.Q.at(i) * x_desired_.at(i));
    if (i < N_) {
      input_costs_[i]->UpdateCoefficients(2 * cost_matrices_.R.at(i),
                                          VectorXd::Zero(n_u_));
    }
  }
}

const std::vector<drake::solvers::QuadraticCost*>& C3::GetTargetCost() {
  return target_costs_;
}

void C3::Solve(const VectorXd& x0) {
  auto start = std::chrono::high_resolution_clock::now();
  // Set the initial state constraint
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

  // Set the initial force constraint
  if (h_is_zero_ == 1) {  // No dependence on u, so just simulate passive system
    drake::solvers::MobyLCPSolver<double> LCPSolver;
    VectorXd lambda0;
    LCPSolver.SolveLcpLemke(lcs_.F()[0], lcs_.E()[0] * x0 + lcs_.c()[0],
                            &lambda0);
    // Force constraints to be updated before every solve if no dependence on u
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

  if (options_.penalize_input_change) {
    for (int i = 0; i < N_; ++i) {
      // Penalize deviation from previous input solution:  input cost is
      // (u-u_prev)' * R * (u-u_prev).
      input_costs_[i]->UpdateCoefficients(
          2 * cost_matrices_.R.at(i),
          -2 * cost_matrices_.R.at(i) * u_sol_->at(i));
    }
  }

  VectorXd delta_init = VectorXd::Zero(n_z_);
  if (options_.delta_option == 1) {
    delta_init.head(n_x_) = x0;
  }
  std::vector<VectorXd> delta(N_, delta_init);
  std::vector<VectorXd> w(N_, VectorXd::Zero(n_z_));
  vector<MatrixXd> G = cost_matrices_.G;

  for (int iter = 0; iter < options_.admm_iter; iter++) {
    ADMMStep(x0, &delta, &w, &G, iter);
  }

  vector<VectorXd> WD(N_, VectorXd::Zero(n_z_));
  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta.at(i) - w.at(i);
  }

  SolveQP(x0, G, WD, delta, options_.admm_iter, true);

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

void C3::ADMMStep(const VectorXd& x0, vector<VectorXd>* delta,
                  vector<VectorXd>* w, vector<MatrixXd>* G,
                  int admm_iteration) {
  vector<VectorXd> WD(N_, VectorXd::Zero(n_z_));

  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta->at(i) - w->at(i);
  }

  vector<VectorXd> z = SolveQP(x0, *G, WD, *delta, admm_iteration, false);

  vector<VectorXd> ZW(N_, VectorXd::Zero(n_z_));
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

void C3::SetInitialGuessQP(const Eigen::VectorXd& x0, int admm_iteration) {
  prog_.SetInitialGuess(x_[0], x0);
  if (!warm_start_ || admm_iteration == 0)
    return;  // No warm start for the first iteration
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
  }
  prog_.SetInitialGuess(x_[N_], warm_start_x_[admm_iteration - 1][N_]);
}

void C3::StoreQPResults(const MathematicalProgramResult& result,
                        int admm_iteration, bool is_final_solve) {
  for (int i = 0; i < N_; ++i) {
    if (is_final_solve) {
      x_sol_->at(i) = result.GetSolution(x_[i]);
      lambda_sol_->at(i) = result.GetSolution(lambda_[i]);
      u_sol_->at(i) = result.GetSolution(u_[i]);
    }
    z_sol_->at(i).segment(0, n_x_) = result.GetSolution(x_[i]);
    z_sol_->at(i).segment(n_x_, n_lambda_) = result.GetSolution(lambda_[i]);
    z_sol_->at(i).segment(n_x_ + n_lambda_, n_u_) = result.GetSolution(u_[i]);
  }

  if (!warm_start_)
    return;  // No warm start, so no need to update warm start parameters
  for (int i = 0; i < N_ + 1; ++i) {
    if (i < N_) {
      warm_start_x_[admm_iteration][i] = result.GetSolution(x_[i]);
      warm_start_lambda_[admm_iteration][i] = result.GetSolution(lambda_[i]);
      warm_start_u_[admm_iteration][i] = result.GetSolution(u_[i]);
    }
    warm_start_x_[admm_iteration][N_] = result.GetSolution(x_[N_]);
  }
}

vector<VectorXd> C3::SolveQP(const VectorXd& x0, const vector<MatrixXd>& G,
                             const vector<VectorXd>& WD,
                             const vector<VectorXd>& delta, int admm_iteration,
                             bool is_final_solve) {
  AddAugmentedCost(G, WD, delta, is_final_solve);
  SetInitialGuessQP(x0, admm_iteration);

  MathematicalProgramResult result = osqp_.Solve(prog_);

  if (!result.is_success()) {
    drake::log()->warn("C3::SolveQP failed to solve the QP with status: {}",
                       result.get_solution_result());
  }

  StoreQPResults(result, admm_iteration, is_final_solve);

  return *z_sol_;
}

void C3::AddAugmentedCost(const vector<MatrixXd>& G, const vector<VectorXd>& WD,
                          const vector<VectorXd>& delta, bool is_final_solve) {
  // Remove previous augmented costs
  for (auto& cost : augmented_costs_) {
    prog_.RemoveCost(cost);
  }
  augmented_costs_.clear();
  // Add or update augmented costs
  for (int i = 0; i < N_; ++i) {
    DRAKE_ASSERT(G.at(i).cols() == WD.at(i).rows());
    augmented_costs_.push_back(prog_.AddQuadraticCost(
        2 * G.at(i), -2 * G.at(i) * WD.at(i), z_.at(i), 1));
  }
}

vector<VectorXd> C3::SolveProjection(const vector<MatrixXd>& U,
                                     vector<VectorXd>& WZ, int admm_iteration) {
  vector<VectorXd> deltaProj(N_, VectorXd::Zero(n_z_));

  if (options_.num_threads > 0) {
    omp_set_dynamic(0);  // Explicitly disable dynamic teams
    omp_set_num_threads(options_.num_threads);  // Set number of threads
    omp_set_nested(0);
    omp_set_schedule(omp_sched_static, 0);
  }

  // clang-format off
#pragma omp parallel for num_threads( \
    options_.num_threads) if (use_parallelization_in_projection_)
  // clang-format on

  for (int i = 0; i < N_; ++i) {
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

void C3::AddLinearConstraint(const Eigen::MatrixXd& A,
                             const VectorXd& lower_bound,
                             const VectorXd& upper_bound,
                             ConstraintVariable constraint) {
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

void C3::AddLinearConstraint(const Eigen::RowVectorXd& A, double lower_bound,
                             double upper_bound,
                             ConstraintVariable constraint) {
  Eigen::VectorXd lb(1);
  lb << lower_bound;
  Eigen::VectorXd ub(1);
  ub << upper_bound;
  AddLinearConstraint(A, lb, ub, constraint);
}

void C3::RemoveConstraints() {
  for (auto& userconstraint : user_constraints_) {
    prog_.RemoveConstraint(userconstraint);
  }
  user_constraints_.clear();
}

const std::vector<LinearConstraintBinding>& C3::GetLinearConstraints() {
  return user_constraints_;
}

}  // namespace c3
