#include <chrono>
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

class C3CartpoleProblem {
  /**
   * This is a fixture we use for testing with GTest and setting up initial
   * testing conditions. It sets up the Cartpole problem which consists of a
   * cart-pole that can interact with soft walls on either side.
   */
public:
  C3CartpoleProblem(float mp = 0.411, float mc = 0.978, float len_p = 0.6,
             float len_com = 0.4267, float d1 = 0.35, float d2 = -0.35,
             float ks = 100, float Ts = 0.01, float g = 9.81) {
    n = 4;
    m = 2;
    k = 1;
    N = 5;

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

    dt = 0.01;

    // Set C3 options
    options.N = N;
    options.dt = dt;
    options.Q = Qinit;
    options.R = Rinit;
    options.G = Ginit;
    options.U = Uinit;
    options.gamma =  1.0;
    options.projection_type = "MIQP";
    options.admm_iter = 10;
    options.rho_scale = 2;
    options.num_threads = 0;
    options.delta_option = 0;
    options.publish_frequency = 50;
    options.solve_time_filter_alpha = 0.0;
    options.end_on_qp_step = true;
    options.use_predicted_x0 = false;
    options.dt = dt;
    options.warm_start = true;

    pSystem = std::make_unique<LCS>(A, B, D, d, E, F, H, c, dt);
    pCost = std::make_unique<C3::CostMatrices>(Q, R, G, U);
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
};
