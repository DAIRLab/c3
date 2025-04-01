#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "core/c3_miqp.h"

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
 * | Solve                  |    -    |
 * | SetOsqpSolverOptions   |    -    |
 * 
 * It also has an E2E test for ensuring the "Solve()" function and other internal functions are working as expected. 
 * However, the E2E takes about 120s at the moment. In order to Disable the test, change the following for the test header.
 * TEST_F(C3CartpoleTest, End2EndCartpoleTest) => TEST_F(C3CartpoleTest, DISABLED_End2EndCartpoleTest)
 * 
 * @todo A coverage report for the cpp files tested.
 */

class C3CartpoleTest : public testing::Test {
  /**
   * This is a fixture we use for testing with GTest and setting up initial testing conditions.
   * It sets up the Cartpole problem which consists of a cart-pole thatcan interact with soft walls on either side.
   */
protected:
  C3CartpoleTest() {
    n = 4;
    m = 2;
    k = 1;
    N = 10;

    float g = 9.81;
    float mp = 0.411;
    float mc = 0.978;
    float len_p = 0.6;
    float len_com = 0.4267;
    float d1 = 0.35;
    float d2 = -0.35;
    float ks = 100;
    float Ts = 0.01;

    /// initial condition(cartpole)
    x0.resize(n);
    x0 << 0.1, 0, 0.3, 0;

    MatrixXd Ainit(n, n);
    Ainit << 0, 0, 1, 0, 0, 0, 0, 1, 0, g * mp / mc, 0, 0, 0,
        (g * (mc + mp)) / (len_com * mc), 0, 0;

    MatrixXd Binit(n, k);
    Binit << 0, 0, 1 / mc, 1 / (len_com * mc);

    MatrixXd Dinit(n, m);
    Dinit << 0, 0, 0, 0, (-1 / mc) + (len_p / (mc * len_com)),
        (1 / mc) - (len_p / (mc * len_com)),
        (-1 / (mc * len_com)) +
            (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
        -((-1 / (mc * len_com)) +
          (len_p * (mc + mp)) / (mc * mp * len_com * len_com));

    MatrixXd Einit(m, n);
    Einit << -1, len_p, 0, 0, 1, -len_p, 0, 0;

    MatrixXd Finit(m, m);
    Finit << 1 / ks, 0, 0, 1 / ks;

    VectorXd cinit(m);
    cinit << d1, -d2;

    VectorXd dinit(n);
    dinit << 0, 0, 0, 0;

    MatrixXd Hinit = MatrixXd::Zero(m, k);

    MatrixXd Qinit(n, n);
    Qinit << 10, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    MatrixXd Rinit(k, k);
    Rinit << 1;

    MatrixXd Ginit(n + m + k, n + m + k);
    Ginit = 0.1 * MatrixXd::Identity(n + m + k, n + m + k);
    Ginit(n + m + k - 1, n + m + k - 1) = 0;

    MatrixXd Uinit(n + m + k, n + m + k);
    Uinit << 1000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0,
        0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0;

    // Use ricatti equation to estimate future cost
    MatrixXd QNinit = drake::math::DiscreteAlgebraicRiccatiEquation(
        Ainit * Ts + MatrixXd::Identity(n, n), Ts * Binit, Qinit, Rinit);

    std::vector<MatrixXd> Qsetup(N + 1, Qinit);
    Qsetup.at(N) = QNinit; // Switch to QNinit, solution of Algebraic Ricatti

    // Initialize LCS for N timesteps
    // Dynamics
    vector<MatrixXd> A(N, Ainit * Ts + MatrixXd::Identity(n, n));
    vector<MatrixXd> B(N, Ts * Binit);
    vector<MatrixXd> D(N, Ts * Dinit);
    vector<VectorXd> d(N, Ts * dinit);

    // Complimentary constraints
    vector<MatrixXd> E(N, Einit);
    vector<MatrixXd> F(N, Finit);
    vector<VectorXd> c(N, cinit);
    vector<MatrixXd> H(N, Hinit);

    // Cost Matrices
    Q = Qsetup;
    vector<MatrixXd> R(N, Rinit);
    vector<MatrixXd> G(N, Ginit);
    vector<MatrixXd> U(N, Uinit);

    // Desired state : Vertically balances
    VectorXd xdesiredinit;
    xdesiredinit = VectorXd::Zero(n);
    xdesired.resize(N + 1, xdesiredinit);

    // Set C3 options
    options.admm_iter = 10;
    options.rho_scale = 2;
    options.num_threads = 0;
    options.delta_option = 0;

    dt = 0.01;
    pSystem = std::make_unique<LCS>(A, B, D, d, E, F, H, c, dt);
    pCost = std::make_unique<C3::CostMatrices>(Q, R, G, U);

    pOpt = std::make_unique<C3MIQP>(*pSystem, *pCost, xdesired, options);
  }

  /// dimensions (n: state dimension, m: complementarity variable dimension, k:
  /// */ input dimension, N: MPC horizon)
  int n, m, k, N;
  double dt;
  /// initial condition(x0), tracking (xdesired)
  VectorXd x0;
  vector<VectorXd> xdesired;
  /// variables (cost, C3)
  vector<MatrixXd> Q;
  /// C3 options
  C3Options options;
  /// Unique pointer to optimizer
  std::unique_ptr<LCS> pSystem;
  std::unique_ptr<C3::CostMatrices> pCost;
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
      public ::testing::WithParamInterface<std::tuple<bool, int, int, int>> {
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
  int constraint_type = std::get<1>(GetParam());
  int constraint_size = std::get<2>(GetParam());
  int num_of_new_constraints = std::get<3>(GetParam());

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

INSTANTIATE_TEST_CASE_P(
    LinearConstraintTests, C3CartpoleTestParameterizedLinearConstraints,
    ::testing::Values(
        std::make_tuple(false, 1, 4, 9), std::make_tuple(false, 2, 1, 10),
        std::make_tuple(false, 3, 2, 10), std::make_tuple(true, 1, 4, 9),
        std::make_tuple(true, 2, 1, 10),
        std::make_tuple(
            true, 3, 2,
            10) // (RowVectorXd/MatrixXd), state, input and force constraints
        ));

// Test if user can remove linear constraints
TEST_F(C3CartpoleTest, RemoveLinearConstraintsTest) {

  RowVectorXd Al = RowVectorXd::Constant(n, 1.0);
  double lb = 0.0;
  double ub = 2.0;

  pOpt->AddLinearConstraint(Al, lb, ub, 1);

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

  std::vector<drake::solvers::QuadraticCost *> target_costs =
      pOpt->GetTargetCost();
  EXPECT_EQ(target_costs.size(), N + 1);

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

  std::vector<drake::solvers::LinearEqualityConstraint *>
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

  std::vector<drake::solvers::LinearEqualityConstraint *>
      pst_dynamic_constraints = pOpt->GetDynamicConstraints();
  for (int i = 0; i < N; ++i) {
    // Linear Equality A matrix should be updated
    MatrixXd pst_Al = pst_dynamic_constraints[i]->GetDenseA();
    EXPECT_EQ(pre_Al[i].isApprox(pst_Al), false);
  }
}

// Test the cartpole example
// This test will take some time to complete ~120s
TEST_F(C3CartpoleTest, End2EndCartpoleTest) {

  /// initialize ADMM variables (delta, w)
  std::vector<VectorXd> delta(N, VectorXd::Zero(n + m + k));
  std::vector<VectorXd> w(N, VectorXd::Zero(n + m + k));

  /// initialize ADMM reset variables (delta, w are reseted to these values)
  std::vector<VectorXd> delta_reset(N, VectorXd::Zero(n + m + k));
  std::vector<VectorXd> w_reset(N, VectorXd::Zero(n + m + k));

  int timesteps = 500; // number of timesteps for the simulation

  /// create state and input arrays
  std::vector<VectorXd> x(timesteps, VectorXd::Zero(n));
  std::vector<VectorXd> input(timesteps, VectorXd::Zero(k));

  x[0] = x0;

  for (int i = 0; i < timesteps - 1; i++) {

    /// reset delta and w (default option)
    delta = delta_reset;
    w = w_reset;

    /// calculate the input given x[i]
    pOpt->Solve(x[i]);
    input[i] = pOpt->GetInputSolution()[0];

    /// simulate the LCS
    x[i + 1] = pSystem->Simulate(x[i], input[i]);
  }
  // Cartpole should be close to center and balancing the pendulum
  ASSERT_EQ(x[timesteps - 1].isZero(0.1), true);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}