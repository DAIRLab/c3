#include <chrono>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "core/test/c3_cartpole_problem.hpp"
#include "core/traj_eval.h"

#include "drake/math/discrete_algebraic_riccati_equation.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using std::vector;

using c3::C3Options;
using c3::traj_eval::TrajectoryEvaluator;

using namespace c3;

/**
 * @brief This file consists of unit tests for all user-facing functions.
 * In bazel, after compilation it can be run using the following command :
 * "bazel test --test_output=all //core:c3_test"
 *
 * The function tested using the unit tests are as follows :
 * | ---------------------- | ------- |
 * | Initialization         |   DONE  |
 * | UpdateTarget           |   DONE  |
 * | GetTargetCost          |   DONE  |
 * | AddLinearConstraint(1) |   DONE  |
 * | AddLinearConstraint(2) |   DONE  |
 * | RemoveConstraints      |   DONE  |
 * | GetLinearConstraints   |   DONE  |
 * | UpdateLCS              |   DONE  |
 * | GetDynamicConstraints  |   DONE  |
 * | UpdateCostMatrix       |   DONE  |
 * | Solve                  |    -    |
 * | SetOsqpSolverOptions   |    -    |
 * | CreatePlaceholderLCS   |   DONE  |
 * | WarmStartSmokeTest     |   DONE  |
 * | # of regression tests  |    2    |
 * | traj_eval              |   DONE  |
 *
 * It also has an E2E test for ensuring the "Solve()" function and other
 * internal functions are working as expected. However, the E2E takes about 120s
 * at the moment. In order to Disable the test, change the following for the
 * test header. TEST_F(C3CartpoleTest, End2EndCartpoleTest) =>
 * TEST_F(C3CartpoleTest, DISABLED_End2EndCartpoleTest)
 *
 * @todo A coverage report for the cpp files tested.
 */

class C3CartpoleTest : public testing::Test, public C3CartpoleProblem {
  /**
   * This is a fixture we use for testing with GTest and setting up initial
   * testing conditions. It sets up the Cartpole problem which consists of a
   * cart-pole thatcan interact with soft walls on either side.
   */
 protected:
  C3CartpoleTest() {
    C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81);
    pOpt = std::make_unique<C3MIQP>(*pSystem, cost, xdesired, options);
  }
  std::unique_ptr<C3MIQP> pOpt;
};

// Test constructor (maintain backward compatibility if constructor changed)
TEST_F(C3CartpoleTest, InitializationTest) {
  // Assert system initialization is working as expected
  ASSERT_NE(nullptr, pOpt);
}

// Test if GetLinearConstraints is working as expected
TEST_F(C3CartpoleTest, LinearConstraintsTest) {
  vector<LinearConstraintBinding> user_constraints;
  ASSERT_NO_THROW({ user_constraints = pOpt->GetLinearConstraints(); });
  ASSERT_EQ(user_constraints.size(), 0);
}

class C3CartpoleTestParameterizedLinearConstraints
    : public C3CartpoleTest,
      public ::testing::WithParamInterface<std::tuple<bool, int, int>> {
  /* Parameterized linear constraints contains
  | --------------------------- | state | input | lambda |
  | constraint_type             | 1     | 2     | 3      |
  | constraint_size             | 4     | 1     | 2      |
  | num_of_expected_constraints | 9     | 10    | 10     | # When N = 10
  */
};

// Test if user can manipulate linear constraints using MatrixXd
TEST_P(C3CartpoleTestParameterizedLinearConstraints, LinearConstraintsTest) {
  bool use_row_vector = std::get<0>(GetParam());
  c3::ConstraintVariable constraint_type =
      static_cast<c3::ConstraintVariable>(std::get<1>(GetParam()));
  int constraint_size = std::get<2>(GetParam());
  int num_of_new_constraints =
      constraint_type == c3::ConstraintVariable::STATE ? N - 1 : N;

  MatrixXd Al;
  if (use_row_vector) {
    Al = RowVectorXd::Constant(constraint_size, 1.0);
    double lb = 0.0;
    double ub = 2.0;
    pOpt->AddLinearConstraint(Al, lb, ub, constraint_type);
  } else {
    Al = MatrixXd::Identity(constraint_size, constraint_size);
    VectorXd lb = Eigen::VectorXd::Constant(constraint_size, 0.0);
    VectorXd ub = Eigen::VectorXd::Constant(constraint_size, 2.0);
    pOpt->AddLinearConstraint(Al, lb, ub, constraint_type);
  }

  vector<LinearConstraintBinding> user_constraints =
      pOpt->GetLinearConstraints();
  // Number of constraints must be N-1 for state and N for input and lambda
  EXPECT_EQ(user_constraints.size(), num_of_new_constraints);

  for (auto constraint : user_constraints) {
    const std::shared_ptr<drake::solvers::LinearConstraint> lc =
        constraint.evaluator();
    // Check A matrix of the constraint
    EXPECT_EQ(Al.isApprox(lc->GetDenseA()), true);
    // Check number of variables in each constraint
    EXPECT_EQ(constraint.GetNumElements(), constraint_size);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LinearConstraintTests, C3CartpoleTestParameterizedLinearConstraints,
    ::testing::Values(std::make_tuple(false, c3::ConstraintVariable::STATE, 4),
                      std::make_tuple(false, c3::ConstraintVariable::INPUT, 1),
                      std::make_tuple(false, c3::ConstraintVariable::FORCE, 2),
                      std::make_tuple(true, c3::ConstraintVariable::STATE, 4),
                      std::make_tuple(true, c3::ConstraintVariable::INPUT, 1),
                      std::make_tuple(true, c3::ConstraintVariable::FORCE,
                                      2)  // (RowVectorXd/MatrixXd), state,
                                          // input and force constraints
                      ));

// Test if user can remove linear constraints
TEST_F(C3CartpoleTest, RemoveLinearConstraintsTest) {
  RowVectorXd Al = RowVectorXd::Constant(n, 1.0);
  double lb = 0.0;
  double ub = 2.0;

  pOpt->AddLinearConstraint(Al, lb, ub, c3::ConstraintVariable::STATE);

  vector<LinearConstraintBinding> user_constraints =
      pOpt->GetLinearConstraints();
  EXPECT_EQ(user_constraints.size(), N - 1);

  // Ensure user constraints are a copy
  user_constraints.clear();
  EXPECT_EQ(user_constraints.size(), 0);
  user_constraints = pOpt->GetLinearConstraints();
  EXPECT_EQ(user_constraints.size(), N - 1);

  // Ensure constraints are removed
  pOpt->RemoveConstraints();
  user_constraints = pOpt->GetLinearConstraints();
  EXPECT_EQ(user_constraints.size(), 0);
}

// Test if user can update the target state
TEST_F(C3CartpoleTest, UpdateTargetTest) {
  // Desired state : random state
  VectorXd xdesiredinit;
  xdesiredinit = Eigen::Vector4d(0.0, 1.0, 2.0, 3.0);
  xdesired.resize(N + 1, xdesiredinit);

  pOpt->UpdateTarget(xdesired);

  vector<drake::solvers::QuadraticCost*> target_costs = pOpt->GetTargetCost();
  EXPECT_EQ(target_costs.size(), N + 1);

  for (int i = 0; i < N + 1; ++i) {
    // Quadratic Q and b cost matrices should be updated
    MatrixXd Qq = cost.Q.at(i) * 2;
    MatrixXd bq = -2 * cost.Q.at(i) * xdesired.at(i);

    EXPECT_EQ(Qq.isApprox(target_costs[i]->Q()), true);
    EXPECT_EQ(bq.isApprox(target_costs[i]->b()), true);
  }
}

// Test if user can update the Cost matrices
TEST_F(C3CartpoleTest, UpdateCostMatrix) {
  vector<MatrixXd> Q(N + 1, MatrixXd::Zero(n, n));
  vector<MatrixXd> R(N, MatrixXd::Zero(k, k));
  vector<MatrixXd> G(N, MatrixXd::Zero(n + m + k, n + m + k));
  vector<MatrixXd> U(N, MatrixXd::Zero(n + m + k, n + m + k));

  C3::CostMatrices new_cost(Q, R, G, U);
  pOpt->UpdateCostMatrices(new_cost);

  const C3::CostMatrices cost_matrices = pOpt->GetCostMatrices();

  // Ensure matrices are updated
  for (int i = 0; i < N + 1; ++i) {
    EXPECT_EQ(cost.Q.at(i).isApprox(cost_matrices.Q.at(i)), false);
    if (i < N) {
      EXPECT_EQ(cost.R.at(i).isApprox(cost_matrices.R.at(i)), false);
      EXPECT_EQ(cost.G.at(i).isApprox(cost_matrices.G.at(i)), false);
      EXPECT_EQ(cost.U.at(i).isApprox(cost_matrices.U.at(i)), false);
    }
  }

  // Ensure target state costs are updated
  vector<drake::solvers::QuadraticCost*> target_costs = pOpt->GetTargetCost();

  for (int i = 0; i < N + 1; ++i) {
    // Quadratic Q and b cost matrices should be updated
    MatrixXd Qq = Q.at(i) * 2;
    MatrixXd bq = -2 * Q.at(i) * xdesired.at(i);

    EXPECT_EQ(Qq.isApprox(target_costs[i]->Q()), true);
    EXPECT_EQ(bq.isApprox(target_costs[i]->b()), true);
  }
}

class C3CartpoleTestParameterizedScalingLCSTest
    : public C3CartpoleTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {};

// [Regression Test] Ensure that LCS in correctly scaled (if required) and used
// in dynamic constraints. This should be true if C3 initialized or LCS is
// updated.
TEST_P(C3CartpoleTestParameterizedScalingLCSTest, ScalingLCSTest) {
  options.scale_lcs = std::get<0>(GetParam());
  bool use_update_lcs = std::get<1>(GetParam());

  C3MIQP optimizer(*pSystem, cost, xdesired, options);
  if (use_update_lcs) {
    optimizer.UpdateLCS(*pSystem);
  }

  if (options.scale_lcs) pSystem->ScaleComplementarityDynamics();

  MatrixXd LinEq = MatrixXd::Zero(n, 2 * n + m + k);
  LinEq.block(0, n + m + k, n, n) = -1 * MatrixXd::Identity(n, n);
  LinEq.block(0, 0, n, n) = pSystem->A().at(0);
  LinEq.block(0, n, n, m) = pSystem->D().at(0);
  LinEq.block(0, n + m, n, k) = pSystem->B().at(0);

  vector<drake::solvers::LinearEqualityConstraint*> dynamic_constraints =
      optimizer.GetDynamicConstraints();
  for (int i = 0; i < N; ++i) {
    // Linear Equality A matrix should be updated
    MatrixXd Ad = dynamic_constraints[i]->GetDenseA();
    EXPECT_EQ(LinEq.isApprox(Ad), true);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ScalingLCSTests, C3CartpoleTestParameterizedScalingLCSTest,
    ::testing::Values(
        std::make_tuple(
            true, true),  // Scale LCS and check initialized dynamic constraints
        std::make_tuple(
            true, false),  // Scale LCS and check updated dynamic constraints
        std::make_tuple(
            false,
            true),  // Don't scale LCS and check initialized dynamic constraints
        std::make_tuple(
            false,
            false)  // Don't scale LCS and check updated dynamic constraints
        ));

// [Regression Test] Related to https://github.com/DAIRLab/c3/issues/6
// Ensure z_sol values not stale due to INFEASIBLE CONSTRAINTS which had occured
// when initial force constraints were incorrectly updated when H is zero.
// Note: this test only works if options.end_on_qp_step = true
TEST_F(C3CartpoleTest, ZSolStaleTest) {
  int timesteps = 5;  // number of timesteps for the simulation

  // Create state and input arrays
  vector<VectorXd> z(timesteps, VectorXd::Zero(n + m + k));
  vector<VectorXd> state(timesteps, VectorXd::Zero(n));
  vector<VectorXd> input(timesteps, VectorXd::Zero(k));

  state[0] << 0.1, -0.5, 0.5, -0.4;  // initial state with contact to right wall

  for (int i = 0; i < timesteps - 1; i++) {
    // Calculate the input given x[i]
    pOpt->Solve(state[i]);
    input[i] = pOpt->GetInputSolution()[0];
    z[i] = pOpt->GetFullSolution()[0];

    // Simulate the LCS

    state[i + 1] = pSystem->Simulate(state[i], input[i]);

    ASSERT_EQ(state[i].segment(0, n).isApprox(z[i].segment(0, n)),
              true);  // Current state should be equal to initial state in
                      // full solution (The assumption is the full solution is
                      // not updated when the QP fails)
  }
}

// Test if CreatePlaceholderLCS works as expected
TEST_F(C3CartpoleTest, CreatePlaceholder) {
  // Create a placeholder LCS object
  LCS placeholder = LCS::CreatePlaceholderLCS(n, k, m, N, dt);
  // Ensure the object is created without any issues
  ASSERT_TRUE(placeholder.HasSameDimensionsAs(*pSystem));
}

// Test if the solver works with warm start enabled (smoke test)
TEST_F(C3CartpoleTest, WarmStartSmokeTest) {
  // Enable warm start option
  options.warm_start = true;
  C3MIQP optimizer(*pSystem, cost, xdesired, options);

  // Solver should not throw when called with warm start
  ASSERT_NO_THROW(optimizer.Solve(x0));
}

class TrajectoryEvaluatorTest : public testing::Test {};

TEST_F(TrajectoryEvaluatorTest, QuadraticCostMatchesManual) {
  vector<VectorXd> x(2, VectorXd::Zero(1));
  vector<VectorXd> u(1, VectorXd::Zero(1));
  vector<VectorXd> lambda(1, VectorXd::Zero(1));
  vector<VectorXd> x_des(2, VectorXd::Zero(1));
  vector<VectorXd> u_des(1, VectorXd::Zero(1));
  vector<VectorXd> lambda_des(1, VectorXd::Zero(1));

  x[0](0) = 1.0;
  x[1](0) = 2.0;
  u[0](0) = 3.0;
  lambda[0](0) = 0.25;
  x_des[0](0) = 0.5;
  x_des[1](0) = 1.5;
  u_des[0](0) = 2.0;
  lambda_des[0](0) = 0.0;

  vector<MatrixXd> Q(2, MatrixXd::Zero(1, 1));
  vector<MatrixXd> R(1, MatrixXd::Zero(1, 1));
  vector<MatrixXd> S(1, MatrixXd::Zero(1, 1));
  Q[0](0, 0) = 2.0;
  Q[1](0, 0) = 4.0;
  R[0](0, 0) = 3.0;
  S[0](0, 0) = 5.0;

  double expected = 0.0;
  expected += 2.0 * 0.5 * 0.5;
  expected += 4.0 * 0.5 * 0.5;
  double actual =
      TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, x_des, Q);
  EXPECT_NEAR(actual, expected, 1e-12);  // State costs

  expected += 3.0 * 1.0 * 1.0;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, x_des, Q, u,
                                                               u_des, R);
  EXPECT_NEAR(actual, expected, 1e-12);  // State and input costs
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, x_des, Q, u,
                                                               u_des[0], R);
  EXPECT_NEAR(actual, expected,
              1e-12);  // State and input costs with repeated desired input

  expected += 5.0 * 0.25 * 0.25;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q, u, u_des, R, lambda, lambda_des, S);
  EXPECT_NEAR(actual, expected, 1e-12);  // State, input, force costs

  // Test with repeated cost matrices across time steps
  expected = 0.0;
  expected += 2.0 * 0.5 * 0.5;
  expected += 2.0 * 0.5 * 0.5;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, x_des, Q[0]);
  EXPECT_NEAR(actual, expected, 1e-12);  // State costs

  expected += 3.0 * 1.0 * 1.0;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, x_des, Q[0],
                                                               u, u_des, R[0]);
  EXPECT_NEAR(actual, expected, 1e-12);  // State and input costs
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q[0], u, u_des[0], R[0]);
  EXPECT_NEAR(actual, expected,
              1e-12);  // State and input costs with repeated desired input

  expected += 5.0 * 0.25 * 0.25;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q[0], u, u_des, R[0], lambda, lambda_des, S[0]);
  EXPECT_NEAR(actual, expected, 1e-12);  // State, input, force costs

  // Test with no desired trajectory (so target is assumed to be origin)
  expected = 0.0;
  expected += 2.0 * 1.0 * 1.0;
  expected += 2.0 * 2.0 * 2.0;
  actual = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x, Q[0]);
  EXPECT_NEAR(actual, expected,
              1e-12);  // State costs with no desired trajectory

  // Test any mismatched dimensions throw errors.
  vector<VectorXd> x_wrong(3, VectorXd::Zero(1));
  ASSERT_ANY_THROW(
      TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(x_wrong, x_des, Q));

  vector<VectorXd> u_wrong(2, VectorXd::Zero(1));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q, u_wrong, u_des, R));

  vector<VectorXd> lambda_wrong(2, VectorXd::Zero(1));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q, u, u_des, R, lambda_wrong, lambda_des, S));

  vector<MatrixXd> Q_wrong_size(2, MatrixXd::Zero(2, 2));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q_wrong_size));

  vector<MatrixXd> R_wrong_size(1, MatrixXd::Zero(2, 2));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q, u, u_des, R_wrong_size));

  vector<MatrixXd> S_wrong_size(1, MatrixXd::Zero(2, 2));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x, x_des, Q, u, u_des, R, lambda, lambda_des, S_wrong_size));
}

TEST_F(TrajectoryEvaluatorTest, SimulatePDControlWithLCSTest) {
  const int n = 4;  // 4 states: 2 positions, 2 velocities
  const int k = 2;  // 2 inputs
  const int m = 1;
  const int N = 3;
  const double dt = 0.1;

  // Simple integrator dynamics: x_next = x + B*u
  MatrixXd A = MatrixXd::Identity(n, n);
  MatrixXd B = MatrixXd::Zero(n, k);
  B(0, 0) = dt;
  B(1, 1) = dt;
  MatrixXd D = MatrixXd::Zero(n, m);
  VectorXd d = VectorXd::Zero(n);
  MatrixXd E = MatrixXd::Zero(m, n);
  MatrixXd F = MatrixXd::Identity(m, m);
  MatrixXd H = MatrixXd::Zero(m, k);
  VectorXd c = VectorXd::Zero(m);

  LCS lcs(A, B, D, d, E, F, H, c, N, dt);

  // Create a planned trajectory
  vector<VectorXd> x_plan(N + 1, VectorXd::Zero(n));
  vector<VectorXd> u_plan(N, VectorXd::Zero(k));

  x_plan[0] << 0.0, 0.0, 0.0, 0.0;
  x_plan[1] << 0.5, 0.5, 0.0, 0.0;
  x_plan[2] << 0.8, 0.8, 0.0, 0.0;
  x_plan[3] << 1.0, 1.0, 0.0, 0.0;

  u_plan[0] << 5.0, 5.0;
  u_plan[1] << 3.0, 3.0;
  u_plan[2] << 2.0, 2.0;

  // PD gains - Kp and Kd must have exactly k non-zero elements
  VectorXd Kp = VectorXd::Zero(n);
  VectorXd Kd = VectorXd::Zero(n);
  Kp(0) = 10.0;
  Kp(1) = 10.0;
  Kd(2) = 1.0;
  Kd(3) = 1.0;

  // Simulate with PD control
  std::pair<vector<VectorXd>, vector<VectorXd>> result =
      TrajectoryEvaluator::SimulatePDControlWithLCS(x_plan, u_plan, Kp, Kd,
                                                    lcs);

  vector<VectorXd> x_sim = result.first;
  vector<VectorXd> u_sim = result.second;

  EXPECT_EQ(x_sim.size(), x_plan.size());
  EXPECT_EQ(u_sim.size(), u_plan.size());
  EXPECT_TRUE(x_sim[0].isApprox(x_plan[0]));

  // Test new overload with no feedforward argument. This should match using an
  // explicit zero feedforward trajectory with use_feedforward = false.
  std::pair<vector<VectorXd>, vector<VectorXd>> result_no_ff =
      TrajectoryEvaluator::SimulatePDControlWithLCS(x_plan, Kp, Kd, lcs);
  std::pair<vector<VectorXd>, vector<VectorXd>> result_zero_ff =
      TrajectoryEvaluator::SimulatePDControlWithLCS(
          x_plan, vector<VectorXd>(N, VectorXd::Zero(k)), Kp, Kd, lcs, false);
  EXPECT_EQ(result_no_ff.first.size(), result_zero_ff.first.size());
  EXPECT_EQ(result_no_ff.second.size(), result_zero_ff.second.size());
  for (int i = 0; i < N + 1; ++i) {
    EXPECT_TRUE(result_no_ff.first[i].isApprox(result_zero_ff.first[i]));
  }
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(result_no_ff.second[i].isApprox(result_zero_ff.second[i]));
  }

  // Test with coarse and fine LCS (fine has 2x time resolution)
  LCS fine_lcs(A, B / 2.0, D, d, E, F, H, c, N * 2, dt / 2);
  std::pair<vector<VectorXd>, vector<VectorXd>> result_fine =
      TrajectoryEvaluator::SimulatePDControlWithLCS(x_plan, u_plan, Kp, Kd, lcs,
                                                    fine_lcs);

  vector<VectorXd> x_sim_fine = result_fine.first;
  vector<VectorXd> u_sim_fine = result_fine.second;

  EXPECT_EQ(x_sim_fine.size(), x_plan.size());
  EXPECT_EQ(u_sim_fine.size(), u_plan.size());
  EXPECT_TRUE(x_sim_fine[0].isApprox(x_plan[0]));

  // Test new coarse/fine overload with no feedforward argument.
  std::pair<vector<VectorXd>, vector<VectorXd>> result_fine_no_ff =
      TrajectoryEvaluator::SimulatePDControlWithLCS(x_plan, Kp, Kd, lcs,
                                                    fine_lcs);
  std::pair<vector<VectorXd>, vector<VectorXd>> result_fine_zero_ff =
      TrajectoryEvaluator::SimulatePDControlWithLCS(
          x_plan, vector<VectorXd>(N, VectorXd::Zero(k)), Kp, Kd, lcs, fine_lcs,
          false);
  EXPECT_EQ(result_fine_no_ff.first.size(), result_fine_zero_ff.first.size());
  EXPECT_EQ(result_fine_no_ff.second.size(), result_fine_zero_ff.second.size());
  for (int i = 0; i < N + 1; ++i) {
    EXPECT_TRUE(
        result_fine_no_ff.first[i].isApprox(result_fine_zero_ff.first[i]));
  }
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(
        result_fine_no_ff.second[i].isApprox(result_fine_zero_ff.second[i]));
  }

  // Test error checking for mismatched dimensions
  VectorXd Kp_wrong_size = VectorXd::Zero(n + 1);
  ASSERT_ANY_THROW(TrajectoryEvaluator::SimulatePDControlWithLCS(
      x_plan, u_plan, Kp_wrong_size, Kd, lcs));

  VectorXd Kd_wrong_size = VectorXd::Zero(n - 1);
  ASSERT_ANY_THROW(TrajectoryEvaluator::SimulatePDControlWithLCS(
      x_plan, u_plan, Kp, Kd_wrong_size, lcs));

  // Test wrong number of non-zero elements (must match k)
  VectorXd Kp_wrong_count = VectorXd::Zero(n);
  Kp_wrong_count(0) = 1.0;  // Only 1 non-zero instead of k=2
  ASSERT_ANY_THROW(TrajectoryEvaluator::SimulatePDControlWithLCS(
      x_plan, u_plan, Kp_wrong_count, Kd, lcs));
}

TEST_F(TrajectoryEvaluatorTest, ZeroOrderHoldAndDownsampleRoundTrip) {
  vector<VectorXd> coarse(2, VectorXd::Zero(2));
  coarse[0] << 1.0, 2.0;
  coarse[1] << 3.0, 4.0;

  const int upsample_rate = 3;
  vector<VectorXd> fine =
      TrajectoryEvaluator::ZeroOrderHoldTrajectory(coarse, upsample_rate);
  EXPECT_EQ(fine.size(), coarse.size() * upsample_rate);
  for (int i = 0; i < upsample_rate; ++i) {
    EXPECT_TRUE(fine[i].isApprox(coarse[0]));
    EXPECT_TRUE(fine[i + upsample_rate].isApprox(coarse[1]));
  }

  vector<VectorXd> downsampled =
      TrajectoryEvaluator::DownsampleTrajectory(fine, upsample_rate);
  EXPECT_EQ(downsampled.size(), coarse.size());
  EXPECT_TRUE(downsampled[0].isApprox(coarse[0]));
  EXPECT_TRUE(downsampled[1].isApprox(coarse[1]));
}

TEST_F(TrajectoryEvaluatorTest, MultiZeroOrderHoldAndDownsampleRoundTrip) {
  vector<VectorXd> x_coarse(3, VectorXd::Zero(2));
  x_coarse[0] << 1.0, 2.0;
  x_coarse[1] << 3.0, 4.0;
  x_coarse[2] << 5.0, 6.0;
  vector<VectorXd> u_coarse(2, VectorXd::Zero(1));
  u_coarse[0] << 7.0;
  u_coarse[1] << 8.0;

  const int upsample_rate = 3;
  std::pair<vector<VectorXd>, vector<VectorXd>> fine =
      TrajectoryEvaluator::ZeroOrderHoldTrajectories(x_coarse, u_coarse,
                                                     upsample_rate);
  vector<VectorXd> x_fine = fine.first;
  vector<VectorXd> u_fine = fine.second;
  EXPECT_EQ(x_fine.size(), (x_coarse.size() - 1) * upsample_rate + 1);
  EXPECT_EQ(u_fine.size(), u_coarse.size() * upsample_rate);
  for (int i = 0; i < upsample_rate; ++i) {
    EXPECT_TRUE(x_fine[i].isApprox(x_coarse[0]));
    EXPECT_TRUE(x_fine[i + upsample_rate].isApprox(x_coarse[1]));
    EXPECT_TRUE(u_fine[i].isApprox(u_coarse[0]));
    EXPECT_TRUE(u_fine[i + upsample_rate].isApprox(u_coarse[1]));
  }
  EXPECT_TRUE(x_fine.back().isApprox(x_coarse.back()));

  std::pair<vector<VectorXd>, vector<VectorXd>> downsampled =
      TrajectoryEvaluator::DownsampleTrajectories(x_fine, u_fine,
                                                  upsample_rate);
  vector<VectorXd> x_downsampled = downsampled.first;
  vector<VectorXd> u_downsampled = downsampled.second;
  EXPECT_EQ(x_downsampled.size(), 3);
  EXPECT_EQ(x_downsampled.size(), x_coarse.size());
  EXPECT_EQ(u_downsampled.size(), 2);
  EXPECT_EQ(u_downsampled.size(), u_coarse.size());
  EXPECT_TRUE(x_downsampled[0].isApprox(x_coarse[0]));
  EXPECT_TRUE(x_downsampled[1].isApprox(x_coarse[1]));
  EXPECT_TRUE(x_downsampled[2].isApprox(x_coarse[2]));
  EXPECT_TRUE(u_downsampled[0].isApprox(u_coarse[0]));
  EXPECT_TRUE(u_downsampled[1].isApprox(u_coarse[1]));

  // Test any mismatched dimensions throw errors.
  vector<VectorXd> x_coarse_wrong(2, VectorXd::Zero(3));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ZeroOrderHoldTrajectories(
      x_coarse_wrong, u_coarse, upsample_rate));
  vector<VectorXd> u_coarse_wrong(3, VectorXd::Zero(2));
  ASSERT_ANY_THROW(TrajectoryEvaluator::ZeroOrderHoldTrajectories(
      x_coarse, u_coarse_wrong, upsample_rate));
  vector<VectorXd> x_fine_wrong(2, VectorXd::Zero(3));
  ASSERT_ANY_THROW(TrajectoryEvaluator::DownsampleTrajectories(
      x_fine_wrong, u_fine, upsample_rate));
  vector<VectorXd> u_fine_wrong(3, VectorXd::Zero(2));
  ASSERT_ANY_THROW(TrajectoryEvaluator::DownsampleTrajectories(
      x_fine, u_fine_wrong, upsample_rate));
}

TEST_F(TrajectoryEvaluatorTest,
       SimulateLCSOverTrajectoryMatchesLinearDynamics) {
  const int n = 2;
  const int k = 2;
  const int m = 1;
  const int N = 2;
  const double dt = 0.1;

  MatrixXd A = MatrixXd::Identity(n, n);
  MatrixXd B = MatrixXd::Identity(n, k);
  MatrixXd D = MatrixXd::Zero(n, m);
  VectorXd d = VectorXd::Zero(n);
  MatrixXd E = MatrixXd::Zero(m, n);
  MatrixXd F = MatrixXd::Identity(m, m);
  MatrixXd H = MatrixXd::Zero(m, k);
  VectorXd c = VectorXd::Zero(m);

  LCS lcs(A, B, D, d, E, F, H, c, N, dt);

  VectorXd x_init = VectorXd::Zero(n);
  x_init << 1.0, 1.0;
  vector<VectorXd> u_plan(N, VectorXd::Zero(k));
  u_plan[0] << 1.0, 0.0;
  u_plan[1] << 0.0, 2.0;

  vector<VectorXd> x_sim =
      TrajectoryEvaluator::SimulateLCSOverTrajectory(x_init, u_plan, lcs);
  ASSERT_EQ(x_sim.size(), static_cast<size_t>(N + 1));

  VectorXd x1_expected = x_init + u_plan[0];
  VectorXd x2_expected = x1_expected + u_plan[1];
  EXPECT_TRUE(x_sim[0].isApprox(x_init));
  EXPECT_TRUE(x_sim[1].isApprox(x1_expected));
  EXPECT_TRUE(x_sim[2].isApprox(x2_expected));

  // Test with a finer LCS (smaller dt)
  LCS finer_lcs(A, B / 10.0, D, d, E, F, H, c, N * 10, dt / 10);
  vector<VectorXd> x_sim_from_finer =
      TrajectoryEvaluator::SimulateLCSOverTrajectory(x_init, u_plan, lcs,
                                                     finer_lcs);
  ASSERT_EQ(x_sim_from_finer.size(), static_cast<size_t>(N + 1));
  EXPECT_TRUE(x_sim_from_finer[0].isApprox(x_init));
  EXPECT_TRUE(x_sim_from_finer[1].isApprox(x1_expected));
  EXPECT_TRUE(x_sim_from_finer[2].isApprox(x2_expected));

  // Test any mismatched dimensions throw errors.
  vector<VectorXd> u_plan_wrong(2, VectorXd::Zero(3));
  ASSERT_ANY_THROW(TrajectoryEvaluator::SimulateLCSOverTrajectory(
      x_init, u_plan_wrong, lcs));
  VectorXd x_init_wrong = VectorXd::Zero(3);
  ASSERT_ANY_THROW(TrajectoryEvaluator::SimulateLCSOverTrajectory(x_init_wrong,
                                                                  u_plan, lcs));
}

template <typename T>
class C3CartpoleTypedTest : public testing::Test, public C3CartpoleProblem {
 protected:
  C3CartpoleTypedTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {
    if (std::is_same<T, C3Plus>::value) UseC3Plus();
    pOpt = std::make_unique<T>(*pSystem, cost, xdesired, options);
  }
  std::unique_ptr<T> pOpt;
};

using projection_types = ::testing::Types<C3QP, C3MIQP, C3Plus>;
TYPED_TEST_SUITE(C3CartpoleTypedTest, projection_types);

// Test if user can update the LCS for the C3 problem
TYPED_TEST(C3CartpoleTypedTest, UpdateLCSTest) {
  c3::C3* pOpt = this->pOpt.get();
  auto dt = this->dt;
  vector<drake::solvers::LinearEqualityConstraint*> pre_dynamic_constraints =
      pOpt->GetDynamicConstraints();
  auto& N = this->N;
  auto n = this->n;
  auto k = this->k;
  auto m = this->m;
  vector<MatrixXd> pre_Al(N);
  for (int i = 0; i < N; ++i) {
    pre_Al[i] = pre_dynamic_constraints[i]->GetDenseA();
  }

  // Dynamics
  vector<MatrixXd> An(N, MatrixXd::Identity(n, n));
  vector<MatrixXd> Bn(N, MatrixXd::Identity(n, k));
  vector<MatrixXd> Dn(N, MatrixXd::Identity(n, m));
  vector<VectorXd> dn(N, VectorXd::Ones(n));

  // Complimentary constraints
  vector<MatrixXd> En(N, MatrixXd::Identity(m, n));
  vector<MatrixXd> Fn(N, MatrixXd::Identity(m, m));
  vector<VectorXd> cn(N, VectorXd::Ones(m));
  vector<MatrixXd> Hn(N, MatrixXd::Identity(m, k));

  LCS TestSystem(An, Bn, Dn, dn, En, Fn, Hn, cn, dt);

  pOpt->UpdateLCS(TestSystem);

  vector<drake::solvers::LinearEqualityConstraint*> pst_dynamic_constraints =
      pOpt->GetDynamicConstraints();
  for (int i = 0; i < N; ++i) {
    // Linear Equality A matrix should be updated
    MatrixXd pst_Al = pst_dynamic_constraints[i]->GetDenseA();
    EXPECT_EQ(pre_Al[i].isApprox(pst_Al), false);
  }
}

// Test the cartpole example
// This test will take some time to complete ~30s
TYPED_TEST(C3CartpoleTypedTest, End2EndCartpoleTest) {
  int timesteps = 1000;  // number of timesteps for the simulation

  /// create state and input arrays
  vector<VectorXd> x(timesteps, VectorXd::Zero(this->n));
  vector<VectorXd> input(timesteps, VectorXd::Zero(this->k));

  x[0] = this->x0;

  int close_to_zero_counter = 0;
  for (int i = 0; i < timesteps - 1; i++) {
    /// calculate the input given x[i]
    this->pOpt->Solve(x[i]);
    input[i] = this->pOpt->GetInputSolution()[0];

    /// simulate the LCS
    x[i + 1] = this->pSystem->Simulate(x[i], input[i]);
    if (x[i + 1].isZero(0.1)) {
      close_to_zero_counter++;
      if (close_to_zero_counter == 30) break;
    } else {
      close_to_zero_counter = 0;
    }
  }
  // Cartpole should be close to center and balancing the pendulum
  ASSERT_EQ(x[timesteps - 1].isZero(0.1), true);
}

TYPED_TEST(C3CartpoleTypedTest, ComputeCost) {
  C3Options options_no_input_change = this->options;
  options_no_input_change.penalize_input_change = false;
  TypeParam optimizer(*this->pSystem, this->cost, this->xdesired,
                      options_no_input_change);

  // Solve one iteration of the problem.
  optimizer.Solve(this->x0);
  vector<VectorXd> x_sol = optimizer.GetStateSolution();
  vector<VectorXd> u_sol = optimizer.GetInputSolution();
  vector<VectorXd> lambda_sol = optimizer.GetForceSolution();

  // The state trajectory excludes the N+1 time step.  Add it to the end via
  // LCS rollout since it contributes to the C3 internal cost optimization.
  const LCS& lcs = optimizer.GetLCS();
  const MatrixXd& A_N = lcs.A().back();
  const MatrixXd& B_N = lcs.B().back();
  const MatrixXd& D_N = lcs.D().back();
  const VectorXd& d_N = lcs.d().back();
  x_sol.push_back(A_N * x_sol.back() + B_N * u_sol.back() +
                  D_N * lambda_sol.back() + d_N);

  // Get the cost matrices and desired state.
  const C3::CostMatrices cost_matrices = optimizer.GetCostMatrices();
  vector<MatrixXd> Q = cost_matrices.Q;
  vector<MatrixXd> R = cost_matrices.R;
  vector<VectorXd> x_des = optimizer.GetDesiredState();
  ASSERT_EQ(x_sol.size(), x_des.size());
  ASSERT_EQ(x_sol.size(), Q.size());
  ASSERT_EQ(u_sol.size(), R.size());
  ASSERT_EQ(x_sol.size(), u_sol.size() + 1);

  // Compute the cost using the TrajectoryEvaluator
  double cost_from_c3_object =
      TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(&optimizer);
  double cost_from_traj = TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(
      x_sol, x_des, Q, u_sol, R);
  EXPECT_NEAR(cost_from_c3_object, cost_from_traj, 1e-12);

  cost_from_c3_object =
      TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(&optimizer, Q, R);
  EXPECT_NEAR(cost_from_c3_object, cost_from_traj, 1e-12);

  cost_from_c3_object =
      TrajectoryEvaluator::ComputeQuadraticTrajectoryCost(&optimizer, Q, R[0]);
  EXPECT_NEAR(cost_from_c3_object, cost_from_traj, 1e-12);

  // NOTE: Don't test the Q[0] version since this example has a time-varying
  // cost matrix.
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
