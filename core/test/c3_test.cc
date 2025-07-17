#include <chrono>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "core/c3_miqp.h"
#include "core/c3_qp.h"
#include "core/test/c3_cartpole_problem.hpp"

#include "drake/math/discrete_algebraic_riccati_equation.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using std::vector;

using c3::C3Options;

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
 * | # of regression tests  |    2    |
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
  std::vector<LinearConstraintBinding> user_constraints;
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

  std::vector<LinearConstraintBinding> user_constraints =
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

  std::vector<LinearConstraintBinding> user_constraints =
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

  std::vector<drake::solvers::QuadraticCost*> target_costs =
      pOpt->GetTargetCost();
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
  std::vector<drake::solvers::QuadraticCost*> target_costs =
      pOpt->GetTargetCost();

  for (int i = 0; i < N + 1; ++i) {
    // Quadratic Q and b cost matrices should be updated
    MatrixXd Qq = Q.at(i) * 2;
    MatrixXd bq = -2 * Q.at(i) * xdesired.at(i);

    EXPECT_EQ(Qq.isApprox(target_costs[i]->Q()), true);
    EXPECT_EQ(bq.isApprox(target_costs[i]->b()), true);
  }
}

// Test if user can update the LCS for the C3 problem
TEST_F(C3CartpoleTest, UpdateLCSTest) {
  std::vector<drake::solvers::LinearEqualityConstraint*>
      pre_dynamic_constraints = pOpt->GetDynamicConstraints();
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

  std::vector<drake::solvers::LinearEqualityConstraint*>
      pst_dynamic_constraints = pOpt->GetDynamicConstraints();
  for (int i = 0; i < N; ++i) {
    // Linear Equality A matrix should be updated
    MatrixXd pst_Al = pst_dynamic_constraints[i]->GetDenseA();
    EXPECT_EQ(pre_Al[i].isApprox(pst_Al), false);
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

  std::vector<drake::solvers::LinearEqualityConstraint*> dynamic_constraints =
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
  std::vector<VectorXd> z(timesteps, VectorXd::Zero(n + m + k));
  std::vector<VectorXd> state(timesteps, VectorXd::Zero(n));
  std::vector<VectorXd> input(timesteps, VectorXd::Zero(k));

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

template <typename T>
class C3CartpoleTypedTest : public testing::Test, public C3CartpoleProblem {
 protected:
  C3CartpoleTypedTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {
    pOpt = std::make_unique<T>(*pSystem, cost, xdesired, options);
  }
  std::unique_ptr<T> pOpt;
};

using projection_types = ::testing::Types<C3QP, C3MIQP>;
TYPED_TEST_SUITE(C3CartpoleTypedTest, projection_types);

// Test the cartpole example
// This test will take some time to complete ~30s
TYPED_TEST(C3CartpoleTypedTest, End2EndCartpoleTest) {
  int timesteps = 1000;  // number of timesteps for the simulation

  /// create state and input arrays
  std::vector<VectorXd> x(timesteps, VectorXd::Zero(this->n));
  std::vector<VectorXd> input(timesteps, VectorXd::Zero(this->k));

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}