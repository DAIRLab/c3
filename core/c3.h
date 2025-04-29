#pragma once

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "lcs.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/osqp_solver.h"

namespace c3 {
  typedef drake::solvers::Binding<drake::solvers::LinearConstraint> LinearConstraintBinding;

class C3 {

 public:

  /*!
   * Cost matrices for the MPC and ADMM  algorithm.
   * Q and R define standard MPC costs, while G and U are used as similarity
   * costs in the ADMM Augmented lagrangian and projection solves, respectively
   */
  struct CostMatrices {
    CostMatrices(const std::vector<Eigen::MatrixXd>& Q,
                 const std::vector<Eigen::MatrixXd>& R,
                 const std::vector<Eigen::MatrixXd>& G,
                 const std::vector<Eigen::MatrixXd>& U);
    std::vector<Eigen::MatrixXd> Q;
    std::vector<Eigen::MatrixXd> R;
    std::vector<Eigen::MatrixXd> G;
    std::vector<Eigen::MatrixXd> U;
  };

  /*!
   * @param LCS system dynamics, defined as an LCS (see lcs.h)
   * @param costs Cost function parameters (see above)
   * @param x_desired reference trajectory, where the each vector is the desired state at the corresponding knot point
   * @param options see c3_options.h
   */
  C3(const LCS& LCS, const CostMatrices& costs,
     const std::vector<Eigen::VectorXd>& x_desired, const C3Options& options);

  virtual ~C3() = default;

  /*!
   * Solve the MPC problem
   * @param x0 The initial state of the system
   */
  void Solve(const Eigen::VectorXd& x0);

  /*!
   * Update the dynamics
   * @param lcs the new LCS
   */
  void UpdateLCS(const LCS& lcs);

  /**
   * @brief Get a vector dynamic constraints.
   * [Aᵢ Dᵢ Bᵢ −1]| xᵢ | = -dᵢ
   *              | λᵢ |
   *              | uᵢ |
   *              |xᵢ₊₁|
   * (Aᵢ, Dᵢ, Bᵢ, dᵢ) are defined by the LCS (Linear Complimentary System). 
   * (xᵢ, λᵢ, uᵢ) are optimization variables for state, force and input respectively
   * Each element of the vector provides the dynamics constraint between the (i+1) and ith timesteps. 
   * A total of (N-1) constraints. 
   * 
   * @return const std::vector<drake::solvers::LinearEqualityConstraint*>& 
   */
  const std::vector<drake::solvers::LinearEqualityConstraint*>& GetDynamicConstraints();

  /*!
   * Update the reference trajectory
   * @param x_des the new reference trajectory
   */
  void UpdateTarget(const std::vector<Eigen::VectorXd>& x_des);

  /**
   * @brief Get the Quadratic Cost which to be minimized at each timestep (N total).
   * xᵢᵀQᵢxᵢ − 2x[d]ᵢQᵢᵀxᵢ
   * xᵢ     : State variable at the ith timestep
   * x[d]ᵢ  : desired state at the ith timestep
   * Qᵢ     : Cost matrix for the ith timestep
   * 
   * @return const std::vector<drake::solvers::QuadraticCost*>& 
   */
  const std::vector<drake::solvers::QuadraticCost*>& GetTargetCost();

  /*!
   * Allow users to add a linear constraint for all timesteps
   * @param A constraint A matrix
   * @param lower_bound
   * @param upper_bound
   * @param constraint 1 for state constraints, 2 for input constraints, and 3
   * for force constraints.
   *
   * TODO (@Brian-Acosta or @ebianchini) make constraint an enum
   *   (or better yet, make this three functions)
   */
  void AddLinearConstraint(const Eigen::MatrixXd& A,
                           const Eigen::VectorXd& lower_bound,
                           const Eigen::VectorXd& upper_bound, int constraint);

  /*!
   * Add a single-row linear constraint
   * See above
   * TODO (@Brian-Acosta or @ebianchini) make constraint an enum
   *  (or better yet, make this three functions)
   */
  void AddLinearConstraint(const Eigen::RowVectorXd& A,
                           double lower_bound, double upper_bound, int constraint);

  /*! Remove all constraints previously added by AddLinearConstraint */
  void RemoveConstraints();

  /**
   * @brief Get a vector of user defined linear constraints. 
   * lb ≼ Axᵢ ≼ ub
   * lb ≼ Auᵢ ≼ ub
   * lb ≼ Aλᵢ ≼ ub
   * xᵢ     : State variable at the ith timestep
   * uᵢ     : Input variable at the ith timestep
   * λᵢ     : Force variable at the ith timestep
   * The vector will consist of m constraints (for State, Input or Force) added for N timesteps.  
   * A total of mN elements in the vector.
   * 
   * @return const std::vector<drake::solvers::Binding<drake::solvers::LinearConstraint>>& 
   */
  const std::vector<LinearConstraintBinding>& GetLinearConstraints();

  void SetOsqpSolverOptions(const drake::solvers::SolverOptions& options) {
    prog_.SetSolverOptions(options);
  }

  std::vector<Eigen::VectorXd> GetFullSolution() { return *z_sol_; }
  std::vector<Eigen::VectorXd> GetStateSolution() { return *x_sol_; }
  std::vector<Eigen::VectorXd> GetForceSolution() { return *lambda_sol_; }
  std::vector<Eigen::VectorXd> GetInputSolution() { return *u_sol_; }
  std::vector<Eigen::VectorXd> GetDualDeltaSolution() { return *delta_sol_; }
  std::vector<Eigen::VectorXd> GetDualWSolution() { return *w_sol_; }

 protected:
  std::vector<std::vector<Eigen::VectorXd>> warm_start_delta_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_binary_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_x_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_lambda_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_u_;
  bool warm_start_;
  const int N_;
  const int n_x_;  // n
  const int n_lambda_;  // m
  const int n_u_;  // k

  /*!
  * Project delta_c onto the LCP constraint.
  * @param U cost weight on the similarity between the pre and post projection
  *          values
  * @param delta_c value to project to the LCP constraint
  * @param E, F, H, c LCS state, force, input, and constant terms
  * @param admm_iteration index of the current ADMM iteration
  * @param warm_start_index index into cache of warm start variables
  * @return
  */
  virtual Eigen::VectorXd SolveSingleProjection(
      const Eigen::MatrixXd& U, const Eigen::VectorXd& delta_c,
      const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& c,
      const int admm_iteration, const int& warm_start_index) = 0;

 private:

  /*!
   * Scales the LCS matrices internally to better condition the problem.
   * This only scales the lambdas.
   */
  void ScaleLCS();

  /*!
   * Solve the projection problem for all time-steps
   * @param U Similarity cost weights for the projection optimization
   * @param WZ A reference to the (z + w) variables
   * @param admm_iteration Index of the current ADMM iteration
   */
  std::vector<Eigen::VectorXd> SolveProjection(
      const std::vector<Eigen::MatrixXd>& U, std::vector<Eigen::VectorXd>& WZ,
      int admm_iteration);

  /*!
 * Solve a single ADMM step
 * @param x0 initial state of the system
 * @param delta copy variables from the previous step
 * @param w dual variables from the previous step
 * @param G augmented lagrangian similarity cost from the previous step
 * @param admm_iteration Index of the current ADMM iteration
 */
  void ADMMStep(const Eigen::VectorXd& x0, std::vector<Eigen::VectorXd>* delta,
                std::vector<Eigen::VectorXd>* w,
                std::vector<Eigen::MatrixXd>* G, int admm_iteration);


  /*!
  * Solve a single QP, without LCP constraints
  * @param x0 The initial state of the system
  * @param G Weights for the augmented lagrangian
  * @param WD A reference to the (w - delta) variables
  * @param admm_iteration Index of the current ADMM iteration
  * @param is_final_solve Whether this is the final ADMM iteration
  */
  std::vector<Eigen::VectorXd> SolveQP(const Eigen::VectorXd& x0,
                                       const std::vector<Eigen::MatrixXd>& G,
                                       const std::vector<Eigen::VectorXd>& WD,
                                       int admm_iteration,
                                       bool is_final_solve = false);

  LCS lcs_;
  double AnDn_ = 1.0; // Scaling factor for lambdas
  const CostMatrices cost_matrices_;
  std::vector<Eigen::VectorXd> x_desired_;
  const C3Options options_;
  double solve_time_ = 0;
  bool h_is_zero_;

  drake::solvers::MathematicalProgram prog_;
  // QP step variables
  drake::solvers::OsqpSolver osqp_;
  std::vector<drake::solvers::VectorXDecisionVariable> x_;
  std::vector<drake::solvers::VectorXDecisionVariable> u_;
  std::vector<drake::solvers::VectorXDecisionVariable> lambda_;

  // QP step constraints
  std::shared_ptr<drake::solvers::LinearEqualityConstraint> initial_state_constraint_;
  std::vector<drake::solvers::LinearEqualityConstraint*> dynamics_constraints_;
  std::vector<LinearConstraintBinding> constraints_;
  std::vector<LinearConstraintBinding> user_constraints_;

  /// Projection step variables are defined outside of the MathematicalProgram
  /// interface

  std::vector<drake::solvers::QuadraticCost*> target_cost_;
  std::vector<drake::solvers::Binding<drake::solvers::QuadraticCost>> costs_;
  std::vector<std::shared_ptr<drake::solvers::QuadraticCost>> input_costs_;

  // Solutions
  std::unique_ptr<std::vector<Eigen::VectorXd>> x_sol_;
  std::unique_ptr<std::vector<Eigen::VectorXd>> lambda_sol_;
  std::unique_ptr<std::vector<Eigen::VectorXd>> u_sol_;

  std::unique_ptr<std::vector<Eigen::VectorXd>> z_sol_;
  std::unique_ptr<std::vector<Eigen::VectorXd>> delta_sol_;
  std::unique_ptr<std::vector<Eigen::VectorXd>> w_sol_;
};

}  // namespace c3
