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

/**
 * @class C3CartpoleProblem
 * @brief A fixture for testing the Cartpole problem using GTest.
 *
 * This class sets up the Cartpole problem, which consists of a cart-pole
 * system that can interact with soft walls on either side. It initializes
 * the system dynamics, constraints, and cost matrices for testing.
 */
class C3CartpoleProblem {
 public:
  /**
   * @brief Constructor to initialize the Cartpole problem.
   *
   * @param mp Mass of the pole.
   * @param mc Mass of the cart.
   * @param len_p Length of the pole.
   * @param len_com Length to the center of mass of the pole.
   * @param d1 Soft wall position on the left.
   * @param d2 Soft wall position on the right.
   * @param ks Spring constant for the soft walls.
   * @param g Gravitational acceleration.
   */
  C3CartpoleProblem(float mp = 0.411, float mc = 0.978, float len_p = 0.6,
                    float len_com = 0.4267, float d1 = 0.35, float d2 = -0.35,
                    float ks = 100, float g = 9.81, int N = 5, float dt = 0.01)
      : mp(mp),
        mc(mc),
        len_p(len_p),
        len_com(len_com),
        d1(d1),
        d2(d2),
        ks(ks),
        g(g),
        N(N),
        dt(dt) {
    // Initialize dimensions
    n = 4;  // State dimension
    m = 2;  // Complementarity variable dimension
    k = 1;  // Input dimension

    // Load controller options from YAML file
    options = drake::yaml::LoadYamlFile<C3Options>(
        "core/test/resources/c3_cartpole_options.yaml");
    float Ts = dt;  // Sampling time for the LCS

    // Initial state
    x0.resize(n);
    x0 << 0.1, -0.5, 0.5, -0.4;

    // System dynamics matrices
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

    // Complementarity constraint matrices
    MatrixXd Einit(m, n);
    Einit << -1, len_p, 0, 0, 1, -len_p, 0, 0;

    MatrixXd Finit(m, m);
    Finit << 1 / ks, 0, 0, 1 / ks;

    VectorXd cinit(m);
    cinit << d1, -d2;

    VectorXd dinit(n);
    dinit << 0, 0, 0, 0;

    MatrixXd Hinit = MatrixXd::Zero(m, k);

    // Initialize LCS for N timesteps
    vector<MatrixXd> A(N, Ainit * Ts + MatrixXd::Identity(n, n));
    vector<MatrixXd> B(N, Ts * Binit);
    vector<MatrixXd> D(N, Ts * Dinit);
    vector<VectorXd> d(N, Ts * dinit);

    vector<MatrixXd> E(N, Einit);
    vector<MatrixXd> F(N, Finit);
    vector<VectorXd> c(N, cinit);
    vector<MatrixXd> H(N, Hinit);

    // Create LCS system
    pSystem = std::make_unique<LCS>(A, B, D, d, E, F, H, c, dt);

    // Use Riccati equation to estimate future cost
    MatrixXd QNinit = drake::math::DiscreteAlgebraicRiccatiEquation(
        Ainit * Ts + MatrixXd::Identity(n, n), Ts * Binit, options.Q,
        options.R);

    // Cost matrices
    std::vector<MatrixXd> Qsetup(N + 1, options.Q);
    Qsetup.at(N) = QNinit;  // Terminal cost matrix

    vector<MatrixXd> Q = Qsetup;
    vector<MatrixXd> R(N, options.R);
    vector<MatrixXd> G(N, options.G);
    vector<MatrixXd> U(N, options.U);

    cost = C3::CostMatrices(Q, R, G, U);

    // Desired state: Vertically balanced
    VectorXd xdesiredinit = VectorXd::Zero(n);
    xdesired.resize(N + 1, xdesiredinit);
  }

  // Cartpole problem parameters
  float mp, mc, len_p, len_com, d1, d2, ks, g;

  // Dimensions (n: state, m: complementarity, k: input, N: MPC horizon)
  int n, m, k, N;
  double dt;

  // Initial condition and desired state
  VectorXd x0;
  vector<VectorXd> xdesired;

  // C3 options
  C3Options options;

  // Unique pointer to LCS system
  std::unique_ptr<LCS> pSystem;

  // Cost matrices object
  C3::CostMatrices cost;
};
