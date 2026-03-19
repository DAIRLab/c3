#include "traj_eval.h"

#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include "lcs.h"
#include "solver_options_io.h"

namespace c3 {
namespace traj_eval {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::SimulatePDControlWithLCS(const vector<VectorXd>& x_plan,
                                              const vector<VectorXd>& u_plan,
                                              const VectorXd& Kp,
                                              const VectorXd& Kd,
                                              const LCS& lcs) {
  CheckLCSAndTrajectoryCompatibility(lcs, x_plan, u_plan);

  int N = lcs.N();
  int n_x = lcs.num_states();
  int n_u = lcs.num_inputs();
  int n_lambda = lcs.num_lambdas();

  // Compute the new trajectory used for cost evaluation.
  vector<VectorXd> x(N + 1, VectorXd::Zero(n_x));
  vector<VectorXd> u(N, VectorXd::Zero(n_u));

  // Ensure the Kp and Kd vectors encode the actuated position and velocity
  // indices within the state vector.
  DRAKE_THROW_UNLESS(Kp.rows() == Kd.rows() && Kp.rows() == n_x);
  DRAKE_THROW_UNLESS((Kp.array() != 0.0).count() == n_u);
  DRAKE_THROW_UNLESS((Kd.array() != 0.0).count() == n_u);
  MatrixXd Kp_mat = MatrixXd::Zero(n_u, n_x);
  MatrixXd Kd_mat = MatrixXd::Zero(n_u, n_x);
  int kp_i = 0;
  int kd_i = 0;
  for (int i = 0; i < n_x; ++i) {
    if (Kp(i) != 0) {
      Kp_mat(kp_i, i) = Kp(i);
      kp_i++;
    }
    if (Kd(i) != 0) {
      Kd_mat(kd_i, i) = Kd(i);
      kd_i++;
    }
  }

  x[0] = x_plan[0];
  for (int i = 0; i < N; i++) {
    u[i] =
        u_plan[i] + Kp_mat * (x_plan[i] - x[i]) + Kd_mat * (x_plan[i] - x[i]);
    x[i + 1] = lcs.Simulate(x[i], u[i]);
  }
  return std::make_pair(x, u);
}

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::SimulatePDControlWithLCS(const vector<VectorXd>& x_plan,
                                              const vector<VectorXd>& u_plan,
                                              const VectorXd& Kp,
                                              const VectorXd& Kd,
                                              const LCS& coarse_lcs,
                                              const LCS& fine_lcs) {
  int upsample_rate = CheckCoarseAndFineLCSCompatibility(coarse_lcs, fine_lcs);

  // Zero-order hold the planned trajectory to match the finer time
  // discretization of the LCS.
  auto [x_plan_finer, u_plan_finer] =
      ZeroOrderHoldTrajectories(x_plan, u_plan, upsample_rate);

  // Do PD control with the finer trajectory and LCS.
  auto [x_sim_fine, u_sim_fine] =
      SimulatePDControlWithLCS(x_plan_finer, u_plan_finer, Kp, Kd, fine_lcs);

  // Downsample the resulting trajectory back to the original time
  // discretization.
  auto [x_sim, u_sim] =
      DownsampleTrajectories(x_sim_fine, u_sim_fine, upsample_rate);

  return std::make_pair(x_sim, u_sim);
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const VectorXd& x_init, const vector<VectorXd>& u_plan, const LCS& lcs,
    const LCSSimulateConfig& config) {
  int N = lcs.N();
  CheckLCSAndTrajectoryCompatibility(lcs, vector<VectorXd>(N + 1, x_init),
                                     u_plan);

  vector<VectorXd> x_sim =
      vector<VectorXd>(N + 1, VectorXd::Zero(x_init.size()));
  x_sim[0] = x_init;
  for (int i = 0; i < N; i++) {
    x_sim[i + 1] = lcs.Simulate(x_sim[i], u_plan[i], config);
  }
  return x_sim;
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const vector<VectorXd>& x_plan, const vector<VectorXd>& u_plan,
    const LCS& lcs, const LCSSimulateConfig& config) {
  return SimulateLCSOverTrajectory(x_plan[0], u_plan, lcs, config);
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const VectorXd& x_init, const vector<VectorXd>& u_plan,
    const LCS& coarse_lcs, const LCS& fine_lcs,
    const LCSSimulateConfig& config) {
  int upsample_rate = CheckCoarseAndFineLCSCompatibility(coarse_lcs, fine_lcs);

  vector<VectorXd> u_plan_finer =
      ZeroOrderHoldTrajectory(u_plan, upsample_rate);
  vector<VectorXd> x_sim_fine =
      SimulateLCSOverTrajectory(x_init, u_plan_finer, fine_lcs, config);
  return DownsampleTrajectories(x_sim_fine, u_plan_finer, upsample_rate).first;
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const vector<VectorXd>& x_plan, const vector<VectorXd>& u_plan,
    const LCS& coarse_lcs, const LCS& fine_lcs,
    const LCSSimulateConfig& config) {
  return SimulateLCSOverTrajectory(x_plan[0], u_plan, coarse_lcs, fine_lcs,
                                   config);
}

vector<VectorXd> TrajectoryEvaluator::ZeroOrderHoldTrajectory(
    const vector<VectorXd>& coarse_traj, const int& upsample_rate) {
  int N = coarse_traj.size();

  // Zero-order hold the planned trajectory to match the finer time
  // discretization of the LCS.
  vector<VectorXd> fine_traj(N * upsample_rate,
                             VectorXd::Zero(coarse_traj[0].size()));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < upsample_rate; j++) {
      fine_traj[i * upsample_rate + j] = coarse_traj[i];
    }
  }
  return fine_traj;
}

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::ZeroOrderHoldTrajectories(const vector<VectorXd>& x_coarse,
                                               const vector<VectorXd>& u_coarse,
                                               const int& upsample_rate) {
  DRAKE_THROW_UNLESS(x_coarse.size() == u_coarse.size() + 1);
  vector<VectorXd> x_fine = ZeroOrderHoldTrajectory(
      vector<VectorXd>(x_coarse.begin(), x_coarse.end() - 1), upsample_rate);
  x_fine.push_back(x_coarse.back());
  vector<VectorXd> u_fine = ZeroOrderHoldTrajectory(u_coarse, upsample_rate);
  return std::make_pair(x_fine, u_fine);
}

vector<VectorXd> TrajectoryEvaluator::DownsampleTrajectory(
    const vector<VectorXd>& fine_traj, const int& upsample_rate) {
  DRAKE_THROW_UNLESS(fine_traj.size() % upsample_rate == 0);
  int N = fine_traj.size() / upsample_rate;

  // Downsample the fine trajectory.
  vector<VectorXd> coarse_traj(N, VectorXd::Zero(fine_traj[0].size()));
  for (int i = 0; i < N; i++) {
    coarse_traj[i] = fine_traj[i * upsample_rate];
  }
  return coarse_traj;
}

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::DownsampleTrajectories(const vector<VectorXd>& x_fine,
                                            const vector<VectorXd>& u_fine,
                                            const int& upsample_rate) {
  DRAKE_THROW_UNLESS(x_fine.size() == u_fine.size() + 1);
  vector<VectorXd> x_coarse = DownsampleTrajectory(
      vector<VectorXd>(x_fine.begin(), x_fine.end() - 1), upsample_rate);
  x_coarse.push_back(x_fine.back());
  vector<VectorXd> u_coarse = DownsampleTrajectory(u_fine, upsample_rate);
  return std::make_pair(x_coarse, u_coarse);
}

int TrajectoryEvaluator::CheckCoarseAndFineLCSCompatibility(
    const LCS& coarse_lcs, const LCS& fine_lcs) {
  DRAKE_THROW_UNLESS(coarse_lcs.num_states() == fine_lcs.num_states());
  DRAKE_THROW_UNLESS(coarse_lcs.num_inputs() == fine_lcs.num_inputs());
  DRAKE_THROW_UNLESS(coarse_lcs.N() <= fine_lcs.N());
  DRAKE_THROW_UNLESS(coarse_lcs.dt() >= fine_lcs.dt());
  DRAKE_THROW_UNLESS(coarse_lcs.N() * coarse_lcs.dt() ==
                     fine_lcs.N() * fine_lcs.dt());
  int upsample_rate = fine_lcs.N() / coarse_lcs.N();
  DRAKE_THROW_UNLESS(fine_lcs.dt() * upsample_rate == coarse_lcs.dt());
  return upsample_rate;
}

void TrajectoryEvaluator::CheckLCSAndTrajectoryCompatibility(
    const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
    const std::vector<Eigen::VectorXd>& u,
    const std::vector<Eigen::VectorXd>& lambda) {
  DRAKE_THROW_UNLESS(lcs.N() == x.size() - 1 && lcs.N() == u.size() &&
                     lcs.N() == lambda.size());
  for (int i = 0; i < lcs.N() + 1; i++) {
    DRAKE_THROW_UNLESS(x[i].size() == lcs.num_states());
    if (i < lcs.N()) {
      DRAKE_THROW_UNLESS(u[i].size() == lcs.num_inputs());
      DRAKE_THROW_UNLESS(lambda[i].size() == lcs.num_lambdas());
    }
  }
}

void TrajectoryEvaluator::CheckLCSAndTrajectoryCompatibility(
    const LCS& lcs, const std::vector<Eigen::VectorXd>& x,
    const std::vector<Eigen::VectorXd>& u) {
  return CheckLCSAndTrajectoryCompatibility(
      lcs, x, u, vector<VectorXd>(lcs.N(), VectorXd::Zero(lcs.num_lambdas())));
}

void TrajectoryEvaluator::CheckLCSAndTrajectoryCompatibility(
    const LCS& lcs, const std::vector<Eigen::VectorXd>& x) {
  return CheckLCSAndTrajectoryCompatibility(
      lcs, x, vector<VectorXd>(lcs.N(), VectorXd::Zero(lcs.num_inputs())),
      vector<VectorXd>(lcs.N(), VectorXd::Zero(lcs.num_lambdas())));
}

}  // namespace traj_eval
}  // namespace c3
