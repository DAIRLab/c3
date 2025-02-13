#include "c3_qp.h"

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "lcs.h"

#include "drake/solvers/mathematical_program.h"
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

C3QP::C3QP(const LCS& LCS, const CostMatrices& costs,
           const vector<VectorXd>& xdesired, const C3Options& options)
    : C3(LCS, costs, xdesired, options) {}

VectorXd C3QP::SolveSingleProjection(const MatrixXd& U, const VectorXd& delta_c,
                                     const MatrixXd& E, const MatrixXd& F,
                                     const MatrixXd& H, const VectorXd& c,
                                     const int admm_iteration,
                                     const int& warm_start_index) {
  drake::solvers::MathematicalProgram prog;
  drake::solvers::SolverOptions solver_options;
  drake::solvers::OsqpSolver osqp_;

  SetC3QPDefaultOsqpOptions(&solver_options);
  prog.SetSolverOptions(solver_options);

  auto xn_ = prog.NewContinuousVariables(n_x_, "x");
  auto ln_ = prog.NewContinuousVariables(n_lambda_, "lambda");
  auto un_ = prog.NewContinuousVariables(n_u_, "u");

  double alpha = 0.01;
  double scaling = 1000;

  MatrixXd EFH(n_lambda_, n_x_ + n_lambda_ + n_u_);
  EFH.block(0, 0, n_lambda_, n_x_) = E / scaling;
  EFH.block(0, n_x_, n_lambda_, n_lambda_) = F / scaling;
  EFH.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = H / scaling;

  prog.AddLinearConstraint(
      EFH, -c / scaling,
      Eigen::VectorXd::Constant(n_lambda_,
                                std::numeric_limits<double>::infinity()),
      {xn_, ln_, un_});

  prog.AddLinearConstraint(
      MatrixXd::Identity(n_lambda_, n_lambda_), VectorXd::Zero(n_lambda_),
      Eigen::VectorXd::Constant(n_lambda_,
                                std::numeric_limits<double>::infinity()),
      ln_);

  MatrixXd New_U = U;
  New_U.block(n_x_, n_x_, n_lambda_, n_lambda_) = alpha * F;

  VectorXd cost_linear = -delta_c.transpose() * New_U;

  prog.AddQuadraticCost(New_U, cost_linear, {xn_, ln_, un_}, 1);
  prog.AddQuadraticCost((1 - alpha) * F, VectorXd::Zero(n_lambda_), ln_, 1);

  MathematicalProgramResult result = osqp_.Solve(prog);

  VectorXd xSol = result.GetSolution(xn_);
  VectorXd lamSol = result.GetSolution(ln_);
  VectorXd uSol = result.GetSolution(un_);

  VectorXd delta_kc = VectorXd::Zero(n_x_ + n_lambda_ + n_u_);
  delta_kc.segment(0, n_x_) = xSol;
  delta_kc.segment(n_x_, n_lambda_) = lamSol;
  delta_kc.segment(n_x_ + n_lambda_, n_u_) = uSol;

  return delta_kc;
}

std::vector<Eigen::VectorXd> C3QP::GetWarmStartDelta() const {
  return warm_start_delta_[0];
}

std::vector<Eigen::VectorXd> C3QP::GetWarmStartBinary() const {
  return warm_start_binary_[0];
}

}  // namespace c3