#include <chrono>
#include <iostream>
#include <string>

#include "core/improved_c3.h"

#include "drake/math/discrete_algebraic_riccati_equation.h"

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using std::vector;

using c3::C3Options;

void init_cartpole(int *n_, int *m_, int *k_, int *N_, vector<MatrixXd> *A_,
                   vector<MatrixXd> *B_, vector<MatrixXd> *D_,
                   vector<VectorXd> *d_, vector<MatrixXd> *E_,
                   vector<MatrixXd> *F_, vector<MatrixXd> *H_,
                   vector<VectorXd> *c_, vector<MatrixXd> *Q_,
                   vector<MatrixXd> *R_, vector<MatrixXd> *G_,
                   vector<MatrixXd> *U_, VectorXd *x0,
                   vector<VectorXd> *xdesired_, C3Options *options);

namespace c3 {

int DoMain(int argc, char *argv[]) {
  /// dimensions (n: state dimension, m: complementarity variable dimension, k:
  /// input dimension, N: MPC horizon)
  int nd, md, kd, Nd;
  /// variables (LCS)
  vector<MatrixXd> Ad, Bd, Dd, Ed, Fd, Hd;
  vector<VectorXd> dd, cd;
  /// initial condition(x0), tracking (xdesired)
  VectorXd x0;
  vector<VectorXd> xdesired;
  /// variables (cost, C3)
  vector<MatrixXd> Qd, Rd, Gd, Ud;
  /// C3 options
  C3Options options;

  init_cartpole(&nd, &md, &kd, &Nd, &Ad, &Bd, &Dd, &dd, &Ed, &Fd, &Hd, &cd, &Qd,
                &Rd, &Gd, &Ud, &x0, &xdesired, &options);

  /// set parameters as const
  const vector<MatrixXd> A = Ad;
  const vector<MatrixXd> B = Bd;
  const vector<MatrixXd> D = Dd;
  const vector<MatrixXd> E = Ed;
  const vector<MatrixXd> F = Fd;
  const vector<MatrixXd> H = Hd;
  const vector<VectorXd> d = dd;
  const vector<VectorXd> c = cd;
  const vector<MatrixXd> Q = Qd;
  const vector<MatrixXd> R = Rd;
  const vector<MatrixXd> G = Gd;
  const vector<MatrixXd> U = Ud;
  const int N = Nd;
  const int n = nd;
  const int m = md;
  const int k = kd;
  const double dt = 0.01;

  LCS system(A, B, D, d, E, F, H, c, dt);
  ImprovedC3::CostMatrices cost(Q, R, G, U);
  ImprovedC3 opt(system, cost, xdesired, options);

  int timesteps = 500; // number of timesteps for the simulation

  /// create state and input arrays
  std::vector<VectorXd> x(timesteps, VectorXd::Zero(n));
  std::vector<VectorXd> input(timesteps, VectorXd::Zero(k));

  /// initialize at x0
  x[0] = x0;

  double total_time = 0;
  for (int i = 0; i < timesteps - 1; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    /// calculate the input given x[i]
    opt.Solve(x[i]);
    input[i] = opt.GetInputSolution()[0];

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Solve time:" << elapsed.count() << std::endl;
    total_time = total_time + elapsed.count();

    /// simulate the LCS
    x[i + 1] = system.Simulate(x[i], input[i]);

    /// print the state
    std::cout << "state: " << x[i + 1] << std::endl;
  }
  std::cout << "Average time: " << total_time / (timesteps - 1) << std::endl;
  return 0;
}

} // namespace c3

int main(int argc, char *argv[]) { return c3::DoMain(argc, argv); }

/// initialize LCS parameters for cartpole
void init_cartpole(int *n_, int *m_, int *k_, int *N_, vector<MatrixXd> *A_,
                   vector<MatrixXd> *B_, vector<MatrixXd> *D_,
                   vector<VectorXd> *d_, vector<MatrixXd> *E_,
                   vector<MatrixXd> *F_, vector<MatrixXd> *H_,
                   vector<VectorXd> *c_, vector<MatrixXd> *Q_,
                   vector<MatrixXd> *R_, vector<MatrixXd> *G_,
                   vector<MatrixXd> *U_, VectorXd *x0,
                   vector<VectorXd> *xdesired_, C3Options *options) {
  int n = 4;
  int m = 2;
  int k = 1;
  int N = 10;

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
  VectorXd x0init(n);
  // x0init << 0.1, 0, 0.3, 0;
  x0init << 0, -0.5, 0.5, -0.4;

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

  MatrixXd Ginit(n + 2 * m + k, n + 2 * m + k);
  Ginit = MatrixXd::Zero(n + 2 * m + k, n + 2 * m + k);
  Ginit.block(n + m + k, n + m + k, m, m) = MatrixXd::Identity(m, m);
  Ginit.block(n, n, m, m) = MatrixXd::Identity(m, m);

  MatrixXd QNinit = drake::math::DiscreteAlgebraicRiccatiEquation(
      Ainit * Ts + MatrixXd::Identity(n, n), Ts * Binit, Qinit, Rinit);

  std::vector<MatrixXd> Qsetup(N + 1, Qinit);
  Qsetup.at(N) = QNinit; // Switch to QNinit, solution of Algebraic Ricatti

  std::vector<MatrixXd> A(N, Ainit * Ts + MatrixXd::Identity(n, n));
  std::vector<MatrixXd> B(N, Ts * Binit);
  std::vector<MatrixXd> D(N, Ts * Dinit);
  std::vector<VectorXd> d(N, Ts * dinit);
  std::vector<MatrixXd> E(N, Einit);
  std::vector<MatrixXd> F(N, Finit);
  std::vector<VectorXd> c(N, cinit);
  std::vector<MatrixXd> H(N, Hinit);
  std::vector<MatrixXd> Q = Qsetup;
  std::vector<MatrixXd> R(N, Rinit);
  std::vector<MatrixXd> G(N, Ginit);

  VectorXd xdesiredinit;
  xdesiredinit = VectorXd::Zero(n);
  std::vector<VectorXd> xdesired(N + 1, xdesiredinit);

  MatrixXd Us(n + 2 * m + k, n + 2 * m + k);
  Us = MatrixXd::Zero(n + 2 * m + k, n + 2 * m + k);
  Us.block(n, n, m, m) = MatrixXd::Identity(m, m);
  Us.block(n + m + k, n + m + k, m, m) = 10000 * MatrixXd::Identity(m, m);

  vector<MatrixXd> U(N, Us);

  C3Options optionsinit;
  optionsinit.admm_iter = 10;
  optionsinit.rho_scale = 2;
  optionsinit.num_threads = 5;
  optionsinit.delta_option = 0;

  *options = optionsinit;
  *x0 = x0init;
  *n_ = n;
  *m_ = m;
  *k_ = k;
  *N_ = N;
  *A_ = A;
  *B_ = B;
  *D_ = D;
  *d_ = d;
  *c_ = c;
  *E_ = E;
  *F_ = F;
  *H_ = H;
  *Q_ = Q;
  *R_ = R;
  *G_ = G;
  *U_ = U;
  *xdesired_ = xdesired;
};
