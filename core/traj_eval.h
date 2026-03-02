#pragma once

#include <vector>

#include <Eigen/Dense>

#include "c3_options.h"
#include "lcs.h"

namespace c3 {
namespace traj_eval {

/// Utility class for trajectory evaluation computations and simulations.
class TrajectoryEvaluator {
 public:
  /// Compute the quadratic cost of a trajectory of states, inputs, and contact
  /// forces, with respect to a desired trajectory and quadratic cost matrices.
  ///   Cost = Sum_{i = 0, ... N-1}
  ///             (x_i-x_des_i)^T Q_i (x_i-x_des_i) +
  ///             (u_i-u_des_i)^T R_i (u_i-u_des_i) +
  ///             (lambda_i-lambda_des_i)^T S_i (lambda_i-lambda_des_i)
  ///         + (x_N-x_des_N)^T Q_N (x_N-x_des_N)
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& lambda,
      const std::vector<Eigen::VectorXd>& x_des,
      const std::vector<Eigen::VectorXd>& u_des,
      const std::vector<Eigen::VectorXd>& lambda_des,
      const std::vector<Eigen::MatrixXd>& Q,
      const std::vector<Eigen::MatrixXd>& R,
      const std::vector<Eigen::MatrixXd>& S);
  /// Special case: cost does not depend on the contact forces.
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& x_des,
      const std::vector<Eigen::VectorXd>& u_des,
      const std::vector<Eigen::MatrixXd>& Q,
      const std::vector<Eigen::MatrixXd>& R);
  /// Special case: cost does not depend on the inputs or contact forces.
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& x_des,
      const std::vector<Eigen::MatrixXd>& Q);
  /// Special case: cost matrices are the same across all time steps.
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& lambda,
      const std::vector<Eigen::VectorXd>& x_des,
      const std::vector<Eigen::VectorXd>& u_des,
      const std::vector<Eigen::VectorXd>& lambda_des, const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R, const Eigen::MatrixXd& S);
  /// Special case: cost does not depend on the contact forces, and cost
  /// matrices are the same across all time steps.
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& x_des,
      const std::vector<Eigen::VectorXd>& u_des, const Eigen::MatrixXd& Q,
      const Eigen::MatrixXd& R);
  /// Special case: cost does not depend on the inputs or contact forces, and
  /// cost matrices are the same across all time steps.
  static double ComputeQuadraticTrajectoryCost(
      const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& x_des, const Eigen::MatrixXd& Q);

  /// From a planned trajectory of states and inputs and sets of Kp and Kd
  /// gains, use an LCS to simulate tracking the plan with PD control (with
  /// feedforward control), and return the resulting trajectory of actual states
  /// and inputs.
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  SimulatePDControlWithLCS(const std::vector<Eigen::VectorXd>& x_plan,
                           const std::vector<Eigen::VectorXd>& u_plan,
                           const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd,
                           const LCS& lcs);
  /// Special case: simulate plans from a coarser LCS with a finer LCS. The
  /// returned trajectory is downsampled back to be compatible with the coarser
  /// LCS. The two LCSs must be compatible with each other, and the state and
  /// input plans must be compatible with the coarser LCS.
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  SimulatePDControlWithLCS(const std::vector<Eigen::VectorXd>& x_plan,
                           const std::vector<Eigen::VectorXd>& u_plan,
                           const Eigen::VectorXd& Kp, const Eigen::VectorXd& Kd,
                           const LCS& coarse_lcs, const LCS& fine_lcs);

  /// Simulate an LCS forward from an initial condition over a trajectory of
  /// inputs, and return the resulting trajectory of states.
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const Eigen::VectorXd& x_init, const std::vector<Eigen::VectorXd>& u_plan,
      const LCS& lcs);
  /// Special case: a full trajectory of states is provided, but only the
  /// initial state is used for simulation.
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const std::vector<Eigen::VectorXd>& x_plan,
      const std::vector<Eigen::VectorXd>& u_plan, const LCS& lcs);
  /// Special case: simulate a trajectory from a coarser LCS with a finer LCS.
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const Eigen::VectorXd& x_init, const std::vector<Eigen::VectorXd>& u_plan,
      const LCS& coarse_lcs, const LCS& fine_lcs);
  /// Special case: simulate a trajectory from a coarser LCS with a finer LCS,
  /// where a full trajectory of states is provided but only the initial state
  /// is used.
  static std::vector<Eigen::VectorXd> SimulateLCSOverTrajectory(
      const std::vector<Eigen::VectorXd>& x_plan,
      const std::vector<Eigen::VectorXd>& u_plan, const LCS& coarse_lcs,
      const LCS& fine_lcs);

  /// Zero order hold a trajectory compatible with one LCS to be compatible with
  /// a different LCS with a finer time discretization. The LCSs must be
  /// compatible with each other, i.e. the finer horizon must be an integer
  /// multiple of the coarser horizon, and dt*N must be the same for both LCSs.
  /// The state trajectory (which was N+1 in length) becomes
  /// (N*timestep_split)+1 in length, and the input trajectory (which was N in
  /// length) becomes N*timestep_split in length.
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  ZeroOrderHoldTrajectories(const std::vector<Eigen::VectorXd>& x_coarse,
                            const std::vector<Eigen::VectorXd>& u_coarse,
                            const int& timestep_split);
  /// Same as above but only one trajectory is provided and processed.
  /// Trajectories of length N become trajectories of length N*timestep_split.
  static std::vector<Eigen::VectorXd> ZeroOrderHoldTrajectory(
      const std::vector<Eigen::VectorXd>& coarse_traj,
      const int& timestep_split);

  /// The reverse of ZeroOrderHoldTrajectoryToFinerLCS: downsample a trajectory
  /// compatible with a finer LCS to be compatible with a coarser LCS. The same
  /// compatibility requirements apply as for ZeroOrderHoldTrajectoryToFinerLCS.
  static std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
  DownsampleTrajectories(const std::vector<Eigen::VectorXd>& x_fine,
                         const std::vector<Eigen::VectorXd>& u_fine,
                         const int& timestep_split);
  /// Same as above but only one trajectory is provided and processed.
  static std::vector<Eigen::VectorXd> DownsampleTrajectory(
      const std::vector<Eigen::VectorXd>& fine_traj, const int& timestep_split);

  /// Check that two LCSs are compatible with each other, i.e. that the finer
  /// horizon is an integer multiple of the coarser horizon, and that dt*N is
  /// the same for both LCSs. Returns the integer multiple by which the finer
  /// horizon is larger than the coarser horizon (i.e. the number of finer time
  /// steps per coarser time step). NOTE: This function does not assert that the
  /// LCSs' lambda dimensions are the same, since it may be desirable to use
  /// different contact pairs for each LCS. The state and input dimensions must
  /// be the same.
  static int CheckCoarseAndFineLCSCompatibility(const LCS& coarse_lcs,
                                                const LCS& fine_lcs);

  /// Check that a trajectory of states, inputs, and contact forces is
  /// compatible with an LCS, i.e. that the trajectory has the correct number of
  /// time steps and that the dimensions of the states, inputs, and contact
  /// forces match the dimensions of the LCS.
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u,
      const std::vector<Eigen::VectorXd>& lambda);
  /// Special case: no lambdas provided.
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
      const std::vector<Eigen::VectorXd>& u);
  /// Special case: no inputs or lambdas provided.
  static void CheckLCSAndTrajectoryCompatibility(
      const LCS& lcs, const std::vector<Eigen::VectorXd>& x);
};

}  // namespace traj_eval
}  // namespace c3
