#pragma once

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "lcs.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/osqp_solver.h"

namespace c3 {
typedef drake::solvers::Binding<drake::solvers::LinearConstraint>
    LinearConstraintBinding;
typedef drake::solvers::Binding<drake::solvers::QuadraticCost>
    QuadraticCostBinding;

enum ConstraintVariable : uint8_t { STATE = 1, INPUT = 2, FORCE = 3 };

class C3 {
 public:
  /*!
   * Cost matrices for the MPC and ADMM  algorithm.
   * Q and R define standard MPC costs, while G and U are used as similarity
   * costs in the ADMM Augmented lagrangian and projection solves, respectively
   */
  struct CostMatrices {
    CostMatrices() = default;
    CostMatrices(const std::vector<Eigen::MatrixXd>& Q,
                 const std::vector<Eigen::MatrixXd>& R,
                 const std::vector<Eigen::MatrixXd>& G,
                 const std::vector<Eigen::MatrixXd>& U);
    bool HasSameDimensionsAs(const CostMatrices& other) const {
      // Check vector and matrix dimensions
      return (Q.size() == other.Q.size() &&
              Q.at(0).rows() == other.Q.at(0).rows() &&
              Q.at(0).cols() == other.Q.at(0).cols() &&
              R.size() == other.R.size() &&
              R.at(0).rows() == other.R.at(0).rows() &&
              R.at(0).cols() == other.R.at(0).cols() &&
              G.size() == other.G.size() &&
              G.at(0).rows() == other.G.at(0).rows() &&
              G.at(0).cols() == other.G.at(0).cols() &&
              U.size() == other.U.size() &&
              U.at(0).rows() == other.U.at(0).rows() &&
              U.at(0).cols() == other.U.at(0).cols());
    }
    std::vector<Eigen::MatrixXd> Q;
    std::vector<Eigen::MatrixXd> R;
    std::vector<Eigen::MatrixXd> G;
    std::vector<Eigen::MatrixXd> U;
  };

  /*!
   * @param LCS system dynamics, defined as an LCS (see lcs.h)
   * @param costs Cost function parameters (see above)
   * @param x_desired reference trajectory, where the each vector is the desired
   * state at the corresponding knot point
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

  /*!cost_matrices_.
   * Update the dynamics
   * @param lcs the new LCS
   */
  virtual void UpdateLCS(const LCS& lcs);

  /**
   * @brief Get a vector dynamic constraints.
   * [Aᵢ Dᵢ Bᵢ −1]| xᵢ | = -dᵢ
   *              | λᵢ |
   *              | uᵢ |
   *              |xᵢ₊₁|
   * (Aᵢ, Dᵢ, Bᵢ, dᵢ) are defined by the LCS (Linear Complimentary System).
   * (xᵢ, λᵢ, uᵢ) are optimization variables for state, force and input
   * respectively Each element of the vector provides the dynamics constraint
   * between the (i+1) and ith timesteps. A total of (N-1) constraints.
   *
   * @return const std::vector<drake::solvers::LinearEqualityConstraint*>&
   */
  const std::vector<drake::solvers::LinearEqualityConstraint*>&
  GetDynamicConstraints();

  /*!
   * Update the reference trajectory
   * @param x_des the new reference trajectory
   */
  void UpdateTarget(const std::vector<Eigen::VectorXd>& x_des);

  /**
   * @brief Updates the provided cost matrices with new or modified values.
   *
   * This function takes a reference to a CostMatrices.
   * It ensures that the target and input costs are recalculated.
   *
   * @param costs A reference to the CostMatrices object to be updated.
   */
  void UpdateCostMatrices(const CostMatrices& costs);

  /**
   * @brief Get the current cost matrices used in the system.
   *
   * This function provides access to the cost matrices currently being used
   * in the computations.
   *
   * @return const CostMatrices& The current cost matrices.
   */
  const CostMatrices& GetCostMatrices() const { return cost_matrices_; }

  /**
   * @brief Get the Quadratic Cost which to be minimized at each timestep (N
   * total). xᵢᵀQᵢxᵢ − 2x[d]ᵢQᵢᵀxᵢ
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
                           const Eigen::VectorXd& upper_bound,
                           ConstraintVariable constraint);

  /*!
   * Add a single-row linear constraint
   * See above
   * TODO (@Brian-Acosta or @ebianchini) make constraint an enum
   *  (or better yet, make this three functions)
   */
  void AddLinearConstraint(const Eigen::RowVectorXd& A, double lower_bound,
                           double upper_bound, ConstraintVariable constraint);

  /*! Remove all constraints previously added by AddLinearConstraint */
  void RemoveConstraints();

  /**
   * @brief Creates cost matrices from the provided C3Options.
   *
   * This function initializes and populates the cost matrices required for
   * computations within the core module using the values defined in C3Options.
   *
   * @param options The C3Options object containing configuration values.
   * @return CostMatrices The initialized cost matrices.
   */
  static CostMatrices CreateCostMatricesFromC3Options(const C3Options& options,
                                                      int N);

  /**
   * @brief Get a vector of user defined linear constraints.
   * lb ≼ Axᵢ ≼ ub
   * lb ≼ Auᵢ ≼ ub
   * lb ≼ Aλᵢ ≼ ub
   * xᵢ     : State variable at the ith timestep
   * uᵢ     : Input variable at the ith timestep
   * λᵢ     : Force variable at the ith timestep
   * The vector will consist of m constraints (for State, Input or Force) added
   * for N timesteps. A total of mN elements in the vector.
   *
   * @return const
   * std::vector<drake::solvers::Binding<drake::solvers::LinearConstraint>>&
   */
  const std::vector<LinearConstraintBinding>& GetLinearConstraints();

  void SetSolverOptions(const drake::solvers::SolverOptions& options) {
    prog_.SetSolverOptions(options);
  }

  std::vector<Eigen::VectorXd> GetFullSolution() { return *z_sol_; }
  std::vector<Eigen::VectorXd> GetStateSolution() { return *x_sol_; }
  std::vector<Eigen::VectorXd> GetForceSolution() { return *lambda_sol_; }
  std::vector<Eigen::VectorXd> GetInputSolution() { return *u_sol_; }
  std::vector<Eigen::VectorXd> GetDualDeltaSolution() { return *delta_sol_; }
  std::vector<Eigen::VectorXd> GetDualWSolution() { return *w_sol_; }

  int GetZSize() const { return n_z_; }

 protected:
  /// @param lcs      Parameters defining the LCS.
  /// @param costs    Cost matrices used in the optimization.
  /// @param x_des    Desired goal state trajectory.
  /// @param options  Options specific to the C3 formulation.
  /// @param z_size   Size of the z vector, which depends on the specific C3
  /// variant.
  ///                 For example:
  ///                   - C3MIQP / C3QP: z = [x, u, lambda]
  ///                   - C3Plus:        z = [x, u, lambda, eta]
  ///
  /// This constructor is intended for internal use only. The public constructor
  /// delegates to this one, passing in an explicitly computed z vector size.
  C3(const LCS& lcs, const CostMatrices& costs,
     const std::vector<Eigen::VectorXd>& x_des, const C3Options& options,
     int z_size);

  std::vector<std::vector<Eigen::VectorXd>> warm_start_delta_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_binary_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_x_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_lambda_;
  std::vector<std::vector<Eigen::VectorXd>> warm_start_u_;
  bool warm_start_;
  const int N_;
  const int n_x_;       // n
  const int n_lambda_;  // m
  const int n_u_;       // k
  const int n_z_;       // (default) n + m + k

  bool use_parallelization_in_projection_ = true;

  /*!
   * Project delta_c onto the LCP constraint.
   * @param U cost weight on the similarity between the pre and post projection
   *          values
   * @param delta_c value to project to the LCP constraint
   * @param E, F, H, c LCS state, force, input, and constant terms
   * @param admm_iteration index of the current ADMM iteration
   * @param warm_start_index index into cache of warm start variables, -1 for no
   * warm start
   * @return
   */
  virtual Eigen::VectorXd SolveSingleProjection(
      const Eigen::MatrixXd& U, const Eigen::VectorXd& delta_c,
      const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& c,
      const int admm_iteration, const int& warm_start_index) = 0;
  /*!
   * Scales the LCS matrices internally to better condition the problem.
   * This only scales the lambdas.
   */
  void ScaleLCS();

  /*!
   * Set the default solver options for the MathematicalProgram
   * This is called in the constructor, and can be overridden by the user
   */
  void SetDefaultSolverOptions();

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
  std::vector<Eigen::VectorXd> SolveQP(
      const Eigen::VectorXd& x0, const std::vector<Eigen::MatrixXd>& G,
      const std::vector<Eigen::VectorXd>& WD,
      const std::vector<Eigen::VectorXd>& delta, int admm_iteration,
      bool is_final_solve = false);

  /*!
   * @brief Add the augmented Lagrangian cost to the QP problem.
   *
   * This function adds (or updates) the quadratic cost terms for the augmented
   * Lagrangian formulation of the ADMM algorithm. The cost has the form:
   *   ||G(z - (w - delta))||^2
   *
   * For C3Plus subclass, this function also handles special scaling of contact
   * forces during the final solve iteration.
   *
   * @param G Vector of augmented Lagrangian weight matrices, one per timestep.
   *          Dimensions: N x (n_z x n_z)
   * @param WD Vector of (w - delta) terms, one per timestep.
   *           Dimensions: N x n_z
   * @param delta Vector of copy variables from the previous ADMM iteration,
   *              one per timestep. Dimensions: N x n_z
   * @param is_final_solve Boolean flag indicating if this is the final QP solve
   *                       after all ADMM iterations complete. Used by subclasses
   *                       to apply special handling (e.g., contact scaling).
   *
   * @note The cost matrices are stored in augmented_costs_ and removed/re-added
   *       on each call to ensure the QP remains well-posed.
   *
   * @see ADMMStep, SolveQP
   */
  virtual void AddAugmentedCost(const std::vector<Eigen::MatrixXd>& G,
                                const std::vector<Eigen::VectorXd>& WD,
                                const std::vector<Eigen::VectorXd>& delta,
                                bool is_final_solve);

  /*!
   * @brief Set the initial guess (warm start) for the QP solver.
   *
   * This function provides Drake's OSQP solver with an initial guess for the
   * decision variables to accelerate convergence. The warm start strategy
   * interpolates between solutions from previous ADMM iterations if available.
   *
   * For non-first ADMM iterations, the function performs linear interpolation
   * between consecutive warm-start solutions based on the solve_time_ and
   * the LCS time step dt. This allows smooth transitions across multiple
   * planning horizons.
   *
   * @param x0 The initial state at the current planning iteration.
   *           Used to initialize x_[0] = x0.
   * @param admm_iteration The current ADMM iteration index (0-based).
   *                       For iteration 0, no warm start is applied.
   *
   * @note Warm starting is only performed if:
   *       - warm_start_ flag is enabled in C3Options
   *       - admm_iteration > 0 (skip first iteration)
   *
   * The interpolation formula used is:
   *   z_guess[i] = (1 - weight) * warm_start[iter-1][i] + weight * warm_start[iter-1][i+1]
   *   where weight = (solve_time_ mod dt) / dt
   *
   * @see SetInitialGuessQP, warm_start_x_, warm_start_lambda_, warm_start_u_
   */
  virtual void SetInitialGuessQP(const Eigen::VectorXd& x0, int admm_iteration);

  /*!
   * @brief Extract and store the QP solution results.
   *
   * This function retrieves the optimized decision variables from the OSQP
   * solver result and stores them in the solution containers (x_sol_,
   * lambda_sol_, u_sol_, z_sol_). Additionally, it updates warm-start caches
   * to accelerate future ADMM iterations.
   *
   * The function handles two scenarios:
   * 1. Intermediate ADMM iterations: Stores results in z_sol_ for the z-update
   *    step and caches them for warm starting.
   * 2. Final QP solve: Also stores individual solutions in x_sol_, lambda_sol_,
   *    and u_sol_ for final output.
   *
   * Solution storage layout in z_sol_[i]:
   *   z_sol_[i] = [x_[i]; lambda_[i]; u_[i]; ...]
   *   where [...] may include additional variables (e.g., eta in C3Plus)
   *
   * @param result The MathematicalProgramResult from the OSQP solver containing
   *               the optimized decision variables.
   * @param admm_iteration The current ADMM iteration index (0-based).
   * @param is_final_solve Boolean flag indicating if this is the final QP solve.
   *                       If true, also populate x_sol_, lambda_sol_, u_sol_.
   *
   * @note For warm starting:
   *       - warm_start_x_[admm_iteration][i] = x_[i] solution
   *       - warm_start_lambda_[admm_iteration][i] = lambda_[i] solution
   *       - warm_start_u_[admm_iteration][i] = u_[i] solution
   *
   * @see SolveQP, z_sol_, x_sol_, lambda_sol_, u_sol_
   */
  virtual void StoreQPResults(
      const drake::solvers::MathematicalProgramResult& result,
      int admm_iteration, bool is_final_solve);

  LCS lcs_;
  double AnDn_ = 1.0;  // Scaling factor for lambdas
  CostMatrices cost_matrices_;
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
  std::vector<drake::solvers::VariableRefList> z_;

  // QP step constraints
  std::shared_ptr<drake::solvers::LinearEqualityConstraint>
      initial_state_constraint_;
  std::shared_ptr<drake::solvers::LinearEqualityConstraint>
      initial_force_constraint_;
  std::vector<drake::solvers::LinearEqualityConstraint*> dynamics_constraints_;
  std::vector<LinearConstraintBinding> user_constraints_;

  /// Projection step variables are defined outside of the MathematicalProgram
  /// interface

  std::vector<drake::solvers::QuadraticCost*> target_costs_;
  std::vector<QuadraticCostBinding> augmented_costs_;
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
