#pragma once

#include <vector>

#include <Eigen/Dense>

#include "src/c3_options.h"
#include "src/lcs.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"

namespace c3 {
class C3 {
 public:
  /// ( State vector dimension = n, contact vector dimension = m, input vector dimension = k )
  /// @param LCS Linear complementarity system ( x_{k+1} = A_k x_k + B_k u_k + D_k lam_k + d_k, 0 <= lam_k perp E_k x_k + F_k lam_k + H_k u_k + c_k >= 0 )
  /// A_k in R^{n times n}, B_k in R^{n times k}, D_k in R^{n times m}, d_k in R^{n}, E_k in R^{m times n}, F_k in R^{m times m}, H_k in R^{m times k}, c_k in R^{m}
  /// @param Q, R Quadratic Cost function parameters ( sum_k x_k^T Q_k x_k + u_k^T R_k u_k )
  /// Q_k in R^{n times n}, R_k in R^{k times k}
  /// @param G, U ADMM Cost parameters ( check (17) and (19) in paper https://arxiv.org/pdf/2304.11259.pdf )
  /// G_k in R^{n+m+k times n+m+k}, U_k in R^{n+m+k times n+m+k}
  /// @param xdesired Goal (desired) state (x_k - xdesired_k)^T Q_k (x_k - xdesired_k)
  /// xdesired_k in R^{n}
  /// @param options (Solver options such as admm iterations, rho scaling, verbose flags/max iter numbers for OSQP and Gurobi, threading options)

  C3(const LCS& LCS, const std::vector<Eigen::MatrixXd>& Q,
     const std::vector<Eigen::MatrixXd>& R,
     const std::vector<Eigen::MatrixXd>& G,
     const std::vector<Eigen::MatrixXd>& U,
     const std::vector<Eigen::VectorXd>& xdesired,
     const C3Options& options,
     const std::vector<Eigen::VectorXd>& warm_start_delta = {},
     const std::vector<Eigen::VectorXd>& warm_start_binary = {},
     const std::vector<Eigen::VectorXd>& warm_start_x_ = {},
     const std::vector<Eigen::VectorXd>& warm_start_lambda_ = {},
     const std::vector<Eigen::VectorXd>& warm_start_u_ = {},
     bool warm_start = false);

  /// Solve the MPC problem
  /// @param x0 The initial state of the system (in R^{n})
  /// @param delta Copy variable (delta_k in R^{n+m+k})
  /// @param w Scaled dual variable (w_k in R^{n+m+k})
  /// @return The first control action to take, u[0]
  Eigen::VectorXd Solve(Eigen::VectorXd& x0,
                        std::vector<Eigen::VectorXd>& delta,
                        std::vector<Eigen::VectorXd>& w);

  /// Solve a single ADMM step
  /// @param x0 The initial state of the system
  /// @param delta The copy variables from the previous step
  /// @param w The scaled dual variables from the previous step
  /// @param G Cost parameter from previous step
  Eigen::VectorXd ADMMStep(Eigen::VectorXd& x0,
                           std::vector<Eigen::VectorXd>* delta,
                           std::vector<Eigen::VectorXd>* w,
                           std::vector<Eigen::MatrixXd>* G);

  /// Solve a single QP
  /// @param x0 The initial state of the system
  /// @param WD (w - delta) variable
  /// @param G Cost parameter from previous step
  std::vector<Eigen::VectorXd> SolveQP(Eigen::VectorXd& x0,
                                       std::vector<Eigen::MatrixXd>& G,
                                       std::vector<Eigen::VectorXd>& WD);

  /// Solve the projection problem for all timesteps
  /// @param WZ (z + w) variable
  /// @param G Cost parameter from previous step
  std::vector<Eigen::VectorXd> SolveProjection(
      std::vector<Eigen::MatrixXd>& G, std::vector<Eigen::VectorXd>& WZ);

  /// allow users to add constraints (adds for all timesteps)
  /// @param A, Lowerbound, Upperbound Lowerbound <= A^T x <= Upperbound
  /// @param constraint inputconstraint, stateconstraint, forceconstraint
  void AddLinearConstraint(Eigen::RowVectorXd& A, double& Lowerbound,
                           double& Upperbound, int& constraint);

  /// allow user to remove all constraints
  void RemoveConstraints();

  /// Get QP warm start
  std::vector<Eigen::VectorXd> GetWarmStartX() const;
  std::vector<Eigen::VectorXd> GetWarmStartLambda() const;
  std::vector<Eigen::VectorXd> GetWarmStartU() const;

  /// Solve a single projection step
  /// @param E, F, H, c LCS parameters
  /// @param U MIQP cost variable
  /// @param delta_c (z + w) variable
  virtual Eigen::VectorXd SolveSingleProjection(const Eigen::MatrixXd& U,
                                                const Eigen::VectorXd& delta_c,
                                                const Eigen::MatrixXd& E,
                                                const Eigen::MatrixXd& F,
                                                const Eigen::MatrixXd& H,
                                                const Eigen::VectorXd& c,
                                                const int& warm_start_index) = 0;

 public:
  const std::vector<Eigen::MatrixXd> A_;
  const std::vector<Eigen::MatrixXd> B_;
  const std::vector<Eigen::MatrixXd> D_;
  const std::vector<Eigen::VectorXd> d_;
  const std::vector<Eigen::MatrixXd> E_;
  const std::vector<Eigen::MatrixXd> F_;
  const std::vector<Eigen::MatrixXd> H_;
  const std::vector<Eigen::VectorXd> c_;
  const std::vector<Eigen::MatrixXd> Q_;
  const std::vector<Eigen::MatrixXd> R_;
  const std::vector<Eigen::MatrixXd> U_;
  const std::vector<Eigen::MatrixXd> G_;
  const std::vector<Eigen::VectorXd> xdesired_;
  const C3Options options_;
  const int N_;
  const int n_;
  const int m_;
  const int k_;
  const bool hflag_;

protected:
  std::vector<Eigen::VectorXd> warm_start_delta_;
  std::vector<Eigen::VectorXd> warm_start_binary_;
  std::vector<Eigen::VectorXd> warm_start_x_;
  std::vector<Eigen::VectorXd> warm_start_lambda_;
  std::vector<Eigen::VectorXd> warm_start_u_;
  bool warm_start_;

 private:
  drake::solvers::MathematicalProgram prog_;
  drake::solvers::SolverOptions OSQPoptions_;
  drake::solvers::OsqpSolver osqp_;
  std::vector<drake::solvers::VectorXDecisionVariable> x_;
  std::vector<drake::solvers::VectorXDecisionVariable> u_;
  std::vector<drake::solvers::VectorXDecisionVariable> lambda_;
  std::vector<drake::solvers::Binding<drake::solvers::QuadraticCost>> costs_;
  std::vector<drake::solvers::Binding<drake::solvers::LinearConstraint>>
      constraints_;
  std::vector<drake::solvers::Binding<drake::solvers::LinearConstraint>>
      userconstraints_;
};

}  // namespace c3
