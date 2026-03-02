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

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& u,
    const vector<VectorXd>& lambda, const vector<VectorXd>& x_des,
    const vector<VectorXd>& u_des, const vector<VectorXd>& lambda_des,
    const vector<MatrixXd>& Q, const vector<MatrixXd>& R,
    const vector<MatrixXd>& S) {
  DRAKE_THROW_UNLESS(x.size() == x_des.size() && x.size() == Q.size());
  DRAKE_THROW_UNLESS(u.size() == u_des.size() && u.size() == R.size());
  DRAKE_THROW_UNLESS(lambda.size() == lambda_des.size() &&
                     lambda.size() == S.size());
  DRAKE_THROW_UNLESS(x.size() - 1 == u.size() && u.size() == lambda.size());

  int N = x.size() - 1;
  int n_x = x[0].size();
  int n_u = u[0].size();
  int n_lambda = lambda[0].size();

  double cost = 0.0;

  for (int i = 0; i < N + 1; i++) {
    DRAKE_THROW_UNLESS(x[i].size() == n_x && x_des[i].size() == n_x);
    DRAKE_THROW_UNLESS(Q[i].rows() == n_x && Q[i].cols() == n_x);
    cost += (x[i] - x_des[i]).transpose() * Q[i] * (x[i] - x_des[i]);

    if (i < N) {
      DRAKE_THROW_UNLESS(u[i].size() == n_u && u_des[i].size() == n_u);
      DRAKE_THROW_UNLESS(R[i].rows() == n_u && R[i].cols() == n_u);
      cost += (u[i] - u_des[i]).transpose() * R[i] * (u[i] - u_des[i]);

      DRAKE_THROW_UNLESS(lambda[i].size() == n_lambda &&
                         lambda_des[i].size() == n_lambda);
      DRAKE_THROW_UNLESS(S[i].rows() == n_lambda && S[i].cols() == n_lambda);
      cost += (lambda[i] - lambda_des[i]).transpose() * S[i] *
              (lambda[i] - lambda_des[i]);
    }
  }

  return cost;
}

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& u,
    const vector<VectorXd>& x_des, const vector<VectorXd>& u_des,
    const vector<MatrixXd>& Q, const vector<MatrixXd>& R) {
  return ComputeQuadraticTrajectoryCost(
      x, u, vector<VectorXd>(u.size(), VectorXd::Zero(0)), x_des, u_des,
      vector<VectorXd>(u.size(), VectorXd::Zero(0)), Q, R,
      vector<MatrixXd>(u.size(), MatrixXd::Zero(0, 0)));
}

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& x_des,
    const vector<MatrixXd>& Q) {
  return ComputeQuadraticTrajectoryCost(
      x, vector<VectorXd>(x.size() - 1, VectorXd::Zero(0)),
      vector<VectorXd>(x.size() - 1, VectorXd::Zero(0)), x_des,
      vector<VectorXd>(x.size() - 1, VectorXd::Zero(0)),
      vector<VectorXd>(x.size() - 1, VectorXd::Zero(0)), Q,
      vector<MatrixXd>(x.size() - 1, MatrixXd::Zero(0, 0)),
      vector<MatrixXd>(x.size() - 1, MatrixXd::Zero(0, 0)));
}

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& u,
    const vector<VectorXd>& lambda, const vector<VectorXd>& x_des,
    const vector<VectorXd>& u_des, const vector<VectorXd>& lambda_des,
    const MatrixXd& Q, const MatrixXd& R, const MatrixXd& S) {
  return ComputeQuadraticTrajectoryCost(
      x, u, lambda, x_des, u_des, lambda_des, vector<MatrixXd>(x.size(), Q),
      vector<MatrixXd>(u.size(), R), vector<MatrixXd>(lambda.size(), S));
}

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& u,
    const vector<VectorXd>& x_des, const vector<VectorXd>& u_des,
    const MatrixXd& Q, const MatrixXd& R) {
  return ComputeQuadraticTrajectoryCost(x, u, x_des, u_des,
                                        vector<MatrixXd>(x.size(), Q),
                                        vector<MatrixXd>(u.size(), R));
}

double TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
    const vector<VectorXd>& x, const vector<VectorXd>& x_des,
    const MatrixXd& Q) {
  return ComputeQuadraticTrajectoryCost(x, x_des,
                                        vector<MatrixXd>(x.size(), Q));
}

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
  int timestep_split = CheckCoarseAndFineLCSCompatibility(coarse_lcs, fine_lcs);

  // Zero-order hold the planned trajectory to match the finer time
  // discretization of the LCS.
  std::pair<vector<VectorXd>, vector<VectorXd>> fine_plans =
      ZeroOrderHoldTrajectories(x_plan, u_plan, timestep_split);
  vector<VectorXd> x_plan_finer = fine_plans.first;
  vector<VectorXd> u_plan_finer = fine_plans.second;

  // Do PD control with the finer trajectory and LCS.
  std::pair<vector<VectorXd>, vector<VectorXd>> fine_sims =
      SimulatePDControlWithLCS(x_plan_finer, u_plan_finer, Kp, Kd, fine_lcs);
  vector<VectorXd> x_sim_fine = fine_sims.first;
  vector<VectorXd> u_sim_fine = fine_sims.second;

  // Downsample the resulting trajectory back to the original time
  // discretization.
  std::pair<vector<VectorXd>, vector<VectorXd>> downsampled_sims =
      DownsampleTrajectories(x_sim_fine, u_sim_fine, timestep_split);
  vector<VectorXd> x_sim = downsampled_sims.first;
  vector<VectorXd> u_sim = downsampled_sims.second;

  return std::make_pair(x_sim, u_sim);
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const VectorXd& x_init, const vector<VectorXd>& u_plan, const LCS& lcs) {
  int N = lcs.N();
  CheckLCSAndTrajectoryCompatibility(lcs, vector<VectorXd>(N + 1, x_init),
                                     u_plan);

  vector<VectorXd> x_sim =
      vector<VectorXd>(N + 1, VectorXd::Zero(x_init.size()));
  x_sim[0] = x_init;
  for (int i = 0; i < N; i++) {
    x_sim[i + 1] = lcs.Simulate(x_sim[i], u_plan[i]);
  }
  return x_sim;
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const vector<VectorXd>& x_plan, const vector<VectorXd>& u_plan,
    const LCS& lcs) {
  return SimulateLCSOverTrajectory(x_plan[0], u_plan, lcs);
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const VectorXd& x_init, const vector<VectorXd>& u_plan,
    const LCS& coarse_lcs, const LCS& fine_lcs) {
  int timestep_split = CheckCoarseAndFineLCSCompatibility(coarse_lcs, fine_lcs);

  vector<VectorXd> u_plan_finer =
      ZeroOrderHoldTrajectory(u_plan, timestep_split);
  vector<VectorXd> x_sim_fine =
      SimulateLCSOverTrajectory(x_init, u_plan_finer, fine_lcs);
  return DownsampleTrajectories(x_sim_fine, u_plan_finer, timestep_split).first;
}

vector<VectorXd> TrajectoryEvaluator::SimulateLCSOverTrajectory(
    const vector<VectorXd>& x_plan, const vector<VectorXd>& u_plan,
    const LCS& coarse_lcs, const LCS& fine_lcs) {
  return SimulateLCSOverTrajectory(x_plan[0], u_plan, coarse_lcs, fine_lcs);
}

vector<VectorXd> TrajectoryEvaluator::ZeroOrderHoldTrajectory(
    const vector<VectorXd>& coarse_traj, const int& timestep_split) {
  int N = coarse_traj.size();

  // Zero-order hold the planned trajectory to match the finer time
  // discretization of the LCS.
  vector<VectorXd> fine_traj(N * timestep_split,
                             VectorXd::Zero(coarse_traj[0].size()));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < timestep_split; j++) {
      fine_traj[i * timestep_split + j] = coarse_traj[i];
    }
  }
  return fine_traj;
}

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::ZeroOrderHoldTrajectories(const vector<VectorXd>& x_coarse,
                                               const vector<VectorXd>& u_coarse,
                                               const int& timestep_split) {
  DRAKE_THROW_UNLESS(x_coarse.size() == u_coarse.size() + 1);
  vector<VectorXd> x_fine = ZeroOrderHoldTrajectory(
      vector<VectorXd>(x_coarse.begin(), x_coarse.end() - 1), timestep_split);
  x_fine.push_back(x_coarse.back());
  vector<VectorXd> u_fine = ZeroOrderHoldTrajectory(u_coarse, timestep_split);
  return std::make_pair(x_fine, u_fine);
}

vector<VectorXd> TrajectoryEvaluator::DownsampleTrajectory(
    const vector<VectorXd>& fine_traj, const int& timestep_split) {
  DRAKE_THROW_UNLESS(fine_traj.size() % timestep_split == 0);
  int N = fine_traj.size() / timestep_split;

  // Downsample the fine trajectory.
  vector<VectorXd> coarse_traj(N, VectorXd::Zero(fine_traj[0].size()));
  for (int i = 0; i < N; i++) {
    coarse_traj[i] = fine_traj[i * timestep_split];
  }
  return coarse_traj;
}

std::pair<vector<VectorXd>, vector<VectorXd>>
TrajectoryEvaluator::DownsampleTrajectories(const vector<VectorXd>& x_fine,
                                            const vector<VectorXd>& u_fine,
                                            const int& timestep_split) {
  DRAKE_THROW_UNLESS(x_fine.size() == u_fine.size() + 1);
  vector<VectorXd> x_coarse = DownsampleTrajectory(
      vector<VectorXd>(x_fine.begin(), x_fine.end() - 1), timestep_split);
  x_coarse.push_back(x_fine.back());
  vector<VectorXd> u_coarse = DownsampleTrajectory(u_fine, timestep_split);
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
  int timestep_split = fine_lcs.N() / coarse_lcs.N();
  DRAKE_THROW_UNLESS(fine_lcs.dt() * timestep_split == coarse_lcs.dt());
  DRAKE_THROW_UNLESS(coarse_lcs.N() * timestep_split == fine_lcs.N());
  return timestep_split;
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
