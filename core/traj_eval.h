#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "c3.h"
#include "c3_options.h"
#include "lcs.h"

#include "drake/common/drake_throw.h"

namespace c3 {
namespace traj_eval {

/// Utility class for trajectory evaluation computations and simulations.
class TrajectoryEvaluator {
 public:
  /**
   * @brief Computes the quadratic cost of a trajectory of vectors with respect
   * to a desired trajectory and quadratic cost matrices.  Can sum multiple
   * costs together by passing in multiple trajectories, desired trajectories,
   * and cost matrices. The total cost is the sum of the individual costs, where
   * each individual cost is computed as:
   *
   *    Cost = Sum_{i = 0, ... N-1}
   *       (data_i-data_des_i)^T CostMatrix_i (data_i-data_des_i)
   *
   * @param data Trajectory of length N of vectors
   * @param data_des Trajectory of length N of desired vectors
   * @param cost_matrices Trajectory of length N of cost matrices.  This can be
   *                      a single cost matrix instead, in which case that cost
   *                      matrix is used for all time steps.
   * ... (optional additional trajectories, desired trajectories, and cost
   *     matrices, whose additional costs are summed together with the first)
   * @return The cost
   *
   * NOTE: To have this arbitrary number of trajectories (implemented via the
   * Rest input argument with variadic templates), the function definition needs
   * to be in the header file, which is why this function is defined here
   * instead of in the .cc file.  To have source code in the .cc file instead,
   * explicit instantiations of every argument pattern are needed, which is more
   * verbose and less flexible than the variadic template approach.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data,
      const std::vector<Eigen::VectorXd>& data_des,
      const std::vector<Eigen::MatrixXd>& cost_matrices, Rest&&... rest) {
    DRAKE_THROW_UNLESS(data.size() == data_des.size());
    DRAKE_THROW_UNLESS(data.size() == cost_matrices.size());

    int n_data = data[0].size();

    double cost = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
      DRAKE_THROW_UNLESS(data[i].size() == n_data);
      DRAKE_THROW_UNLESS(data_des[i].size() == n_data);
      DRAKE_THROW_UNLESS(cost_matrices[i].rows() == n_data);
      DRAKE_THROW_UNLESS(cost_matrices[i].cols() == n_data);

      cost += (data[i] - data_des[i]).transpose() * cost_matrices[i] *
              (data[i] - data_des[i]);
    }

    return cost + ComputeQuadraticTrajectoryCost(std::forward<Rest>(rest)...);
  }
  /** @brief Special case: the same cost matrix is to be used throughout the
   * trajectory so only one is provided.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data,
      const std::vector<Eigen::VectorXd>& data_des,
      const Eigen::MatrixXd& cost_matrix, Rest&&... rest) {
    return ComputeQuadraticTrajectoryCost(
        data, data_des, std::vector<Eigen::MatrixXd>(data.size(), cost_matrix),
        std::forward<Rest>(rest)...);
  }
  /** @brief Special case: the same desired data is to be used throughout the
   * trajectory so only one is provided.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data, const Eigen::VectorXd& data_des,
      const std::vector<Eigen::MatrixXd>& cost_matrices, Rest&&... rest) {
    return ComputeQuadraticTrajectoryCost(
        data, std::vector<Eigen::VectorXd>(data.size(), data_des),
        cost_matrices, std::forward<Rest>(rest)...);
  }
  /** @brief Special case: the same desired data and the same cost matrix is to
   * be used throughout the trajectory so only one each is provided.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data, const Eigen::VectorXd& data_des,
      const Eigen::MatrixXd& cost_matrix, Rest&&... rest) {
    return ComputeQuadraticTrajectoryCost(
        data, std::vector<Eigen::VectorXd>(data.size(), data_des),
        std::vector<Eigen::MatrixXd>(data.size(), cost_matrix),
        std::forward<Rest>(rest)...);
  }
  /** @brief Special case: no desired data is provided, so it is assumed the
   * desired data is zero.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data,
      const std::vector<Eigen::MatrixXd>& cost_matrices, Rest&&... rest) {
    return ComputeQuadraticTrajectoryCost(
        data,
        std::vector<Eigen::VectorXd>(data.size(),
                                     Eigen::VectorXd::Zero(data[0].size())),
        cost_matrices, std::forward<Rest>(rest)...);
  }
  /** @brief Special case: no desired data is provided and the same cost matrix
   * is to be used throughout the trajectory, so it is assumed the desired data
   * is zero and only one cost matrix is provided.
   */
  template <typename... Rest>
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& data,
      const Eigen::MatrixXd& cost_matrix, Rest&&... rest) {
    return ComputeQuadraticTrajectoryCost(
        data,
        std::vector<Eigen::VectorXd>(data.size(),
                                     Eigen::VectorXd::Zero(data[0].size())),
        std::vector<Eigen::MatrixXd>(data.size(), cost_matrix),
        std::forward<Rest>(rest)...);
  }
  /** @brief Special case: a C3 object is the input argument, and the cost is to
   * be computed based on the trajectories contained in the C3 object and the
   * cost matrices contained in the C3 object for state and input costs.
   *
   * NOTE: This can only handle u^T R u input costs, not (u-u_prev)^T R
   * (u-u_prev) input costs, since the C3 object does not contain u_prev
   * trajectories. If the C3 object is configured to compute input costs based
   * on (u-u_prev), this function throws an error.
   */
  static double ComputeQuadraticTrajectoryCost(C3* c3) {
    // Get the cost matrices.
    std::vector<Eigen::MatrixXd> Q = c3->GetCostMatrices().Q;
    std::vector<Eigen::MatrixXd> R = c3->GetCostMatrices().R;

    return ComputeQuadraticTrajectoryCost(c3, Q, R);
  }
  /** @brief Special case: a C3 object and different cost matrices are the input
   * arguments.
   *
   * NOTE: This can only handle u^T R u input costs, not (u-u_prev)^T R
   * (u-u_prev) input costs, since the C3 object does not contain u_prev
   * trajectories. If the C3 object is configured to compute input costs based
   * on (u-u_prev), this function throws an error.
   */
  static double ComputeQuadraticTrajectoryCost(
      C3* c3, const std::vector<Eigen::MatrixXd>& Q,
      const std::vector<Eigen::MatrixXd>& R) {
    DRAKE_THROW_UNLESS(c3->GetPenalizeInputChange() == false);

    // Get the trajectories (solved for state and input, and desired for state).
    std::vector<Eigen::VectorXd> x = c3->GetStateSolution();
    std::vector<Eigen::VectorXd> u = c3->GetInputSolution();
    std::vector<Eigen::VectorXd> lambda = c3->GetForceSolution();
    std::vector<Eigen::VectorXd> x_des = c3->GetDesiredState();

    // The state trajectory from GetStateSolution excludes the N+1 time step.
    // Add it to the end via LCS rollout since it contributes to the C3 internal
    // cost optimization.
    const LCS& lcs = c3->GetLCS();
    Eigen::MatrixXd A_N = lcs.A().back();
    Eigen::MatrixXd B_N = lcs.B().back();
    Eigen::MatrixXd D_N = lcs.D().back();
    Eigen::VectorXd d_N = lcs.d().back();
    x.push_back(A_N * x.back() + B_N * u.back() + D_N * lambda.back() + d_N);

    DRAKE_THROW_UNLESS(x_des.size() == x.size());
    DRAKE_THROW_UNLESS(x.size() == u.size() + 1);
    DRAKE_THROW_UNLESS(Q.size() == x.size());
    DRAKE_THROW_UNLESS(R.size() == u.size());

    return ComputeQuadraticTrajectoryCost(x, x_des, Q, u, R);
  }
  /** @brief Special cases (3x): a C3 object and singular cost matrices (either
   * Q, or R, or both) are the input arguments. */
  static double ComputeQuadraticTrajectoryCost(
      C3* c3, const Eigen::MatrixXd& Q, const std::vector<Eigen::MatrixXd>& R) {
    return ComputeQuadraticTrajectoryCost(
        c3, std::vector<Eigen::MatrixXd>(c3->GetStateSolution().size() + 1, Q),
        R);
  }
  static double ComputeQuadraticTrajectoryCost(
      C3* c3, const std::vector<Eigen::MatrixXd>& Q, const Eigen::MatrixXd& R) {
    return ComputeQuadraticTrajectoryCost(
        c3, Q, std::vector<Eigen::MatrixXd>(c3->GetInputSolution().size(), R));
  }
  static double ComputeQuadraticTrajectoryCost(C3* c3, const Eigen::MatrixXd& Q,
                                               const Eigen::MatrixXd& R) {
    return ComputeQuadraticTrajectoryCost(
        c3, std::vector<Eigen::MatrixXd>(c3->GetStateSolution().size() + 1, Q),
        std::vector<Eigen::MatrixXd>(c3->GetInputSolution().size(), R));
  }

  /**
   * @brief From a planned trajectory of states and inputs and sets of Kp
   * and Kd gains, use an LCS to simulate tracking the plan with PD control
   * (with feedforward control), and return the resulting trajectory of
   * actual states and inputs.
   *
   * @param x_plan Planned trajectory of states (length N+1)
   * @param u_plan Planned trajectory of inputs (length N)
   * @param Kp Proportional gains (length n_x, with exactly k non-zero
   * entries)
   * @param Kd Derivative gains (length n_x, with exactly k non-zero
   * entries)
   * @param lcs LCS system to simulate
   * @return Pair of (simulated states, simulated inputs)
   */
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  SimulatePDControlWithLCS(const std::vector<Eigen::VectorXd>& x_plan,
                           const std::vector<Eigen::VectorXd>& u_plan,
                           const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd,
                           const LCS& lcs);
  /**
   * @brief Special case: simulate plans from a coarser LCS with a finer LCS.
   * The returned trajectory is downsampled back to be compatible with the
   * coarser LCS. The two LCSs must be compatible with each other, and the state
   * and input plans must be compatible with the coarser LCS.
   */
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  SimulatePDControlWithLCS(const std::vector<Eigen::VectorXd>& x_plan,
                           const std::vector<Eigen::VectorXd>& u_plan,
                           const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd,
                           const LCS& coarse_lcs, const LCS& fine_lcs);

  /**
   * @brief Simulate an LCS forward from an initial condition over a trajectory
   * of inputs, and return the resulting trajectory of states.
   *
   * @param x_init Initial state
   * @param u_plan Trajectory of inputs (length N)
   * @param lcs LCS system to simulate
   * @param config Configuration for simulating the LCS
   * @return Trajectory of simulated states (length N+1)
   */
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const Eigen::VectorXd& x_init, const std::vector<Eigen::VectorXd>& u_plan,
      const LCS& lcs, const LCSSimulateConfig& config = LCSSimulateConfig());
  /** @brief Special case: a full trajectory of states is provided, but only the
   * initial state is used for simulation. */
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const std::vector<Eigen::VectorXd>& x_plan,
      const std::vector<Eigen::VectorXd>& u_plan, const LCS& lcs,
      const LCSSimulateConfig& config = LCSSimulateConfig());
  /**
   * @brief Special case: simulate plans from a coarser LCS with a finer LCS.
   * The returned trajectory is downsampled back to be compatible with the
   * coarser LCS. The two LCSs must be compatible with each other, and the input
   * plan must be compatible with the coarser LCS.
   */
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const Eigen::VectorXd& x_init, const std::vector<Eigen::VectorXd>& u_plan,
      const LCS& coarse_lcs, const LCS& fine_lcs,
      const LCSSimulateConfig& config = LCSSimulateConfig());
  /**
   * @brief Special case: simulate a trajectory from a coarser LCS with a finer
   * LCS, where a full trajectory of states is provided but only the initial
   * state is used.
   */
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const std::vector<Eigen::VectorXd>& x_plan,
      const std::vector<Eigen::VectorXd>& u_plan, const LCS& coarse_lcs,
      const LCS& fine_lcs,
      const LCSSimulateConfig& config = LCSSimulateConfig());

  /**
   * @brief Zero order hold a trajectory compatible with one LCS to be
   * compatible with a different LCS with a finer time discretization. The LCSs
   * must be compatible with each other, i.e. the finer horizon must be an
   * integer multiple of the coarser horizon, and dt*N must be the same for both
   * LCSs. The state trajectory (which was N+1 in length) becomes
   * (N*upsample_rate)+1 in length, and the input trajectory (which was N in
   * length) becomes N*upsample_rate in length.
   *
   * @param x_coarse Coarse state trajectory (length N+1)
   * @param u_coarse Coarse input trajectory (length N)
   * @param upsample_rate Integer multiple for time discretization
   * @return Pair of 1) fine state trajectory (length (N*upsample_rate)+1) and
   * 2) fine input trajectory (length N*upsample_rate)
   */
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  ZeroOrderHoldTrajectories(const std::vector<Eigen::VectorXd>& x_coarse,
                            const std::vector<Eigen::VectorXd>& u_coarse,
                            const int& upsample_rate);
  /**
   * @brief Same as above but only one trajectory is provided and processed.
   * Trajectories of length N become trajectories of length N*upsample_rate.
   */
  static std::vector<Eigen::VectorXd> ZeroOrderHoldTrajectory(
      const std::vector<Eigen::VectorXd>& coarse_traj,
      const int& upsample_rate);

  /**
   * @brief The reverse of ZeroOrderHoldTrajectoryToFinerLCS: downsample a
   * trajectory compatible with a finer LCS to be compatible with a coarser LCS.
   * The same compatibility requirements apply as for
   * ZeroOrderHoldTrajectoryToFinerLCS.
   *
   * @param x_fine Fine state trajectory (length (N*upsample_rate)+1)
   * @param u_fine Fine input trajectory (length N*upsample_rate)
   * @param upsample_rate Integer multiple for time discretization
   * @return Pair of 1) coarse state trajectory (length N+1) and 2) coarse input
   * trajectory (length N)
   */
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  DownsampleTrajectories(const std::vector<Eigen::VectorXd>& x_fine,
                         const std::vector<Eigen::VectorXd>& u_fine,
                         const int& upsample_rate);
  /** @brief Same as above but only one trajectory is provided and processed. */
  static std::vector<Eigen::VectorXd> DownsampleTrajectory(
      const std::vector<Eigen::VectorXd>& fine_traj, const int& upsample_rate);

  /**
   * @brief Check that two LCSs are compatible with each other, i.e. that the
   * finer horizon is an integer multiple of the coarser horizon, and that dt*N
   * is the same for both LCSs. Returns the integer multiple by which the finer
   * horizon is larger than the coarser horizon (i.e. the number of finer time
   * steps per coarser time step). NOTE: This function does not assert that the
   * LCSs' lambda dimensions are the same, since it may be desirable to use
   * different contact pairs for each LCS. The state and input dimensions must
   * be the same.
   *
   * @param coarse_lcs LCS with coarser time discretization
   * @param fine_lcs LCS with finer time discretization
   * @return Integer multiple by which fine_lcs.N() is larger than
   * coarse_lcs.N()
   */
  static int CheckCoarseAndFineLCSCompatibility(const LCS& coarse_lcs,
                                                const LCS& fine_lcs);

  /**
   * @brief Check that a trajectory of states, inputs, and contact forces is
   * compatible with an LCS, i.e. that the trajectory has the correct number of
   * time steps and that the dimensions of the states, inputs, and contact
   * forces match the dimensions of the LCS.
   *
   * @param lcs LCS system
   * @param x State trajectory (length N+1)
   * @param u Input trajectory (length N)
   * @param lambda Contact force trajectory (length N)
   */
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& lambda);
  /** @brief Special case: no lambdas provided. */
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u);
  /** @brief Special case: no inputs or lambdas provided. */
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x);

 private:
  /** @brief Special case of ComputeQuadraticTrajectoryCost: no trajectory to
   * evaluate. Required for variadic template recursion base case, but should
   * not be called externally with no input arguments.
   */
  static double ComputeQuadraticTrajectoryCost() { return 0.0; }
};

}  // namespace traj_eval
}  // namespace c3
