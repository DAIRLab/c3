#include "lcs.h"

#include "drake/solvers/moby_lcp_solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace c3 {

LCS::LCS(const vector<MatrixXd>& A, const vector<MatrixXd>& B,
         const vector<MatrixXd>& D, const vector<VectorXd>& d,
         const vector<MatrixXd>& E, const vector<MatrixXd>& F,
         const vector<MatrixXd>& H, const vector<VectorXd>& c, double dt)
    : A_(A),
      B_(B),
      D_(D),
      d_(d),
      E_(E),
      F_(F),
      H_(H),
      c_(c),
      N_(A_.size()),
      dt_(dt),
      n_(A_[0].rows()),
      m_(D_[0].cols()),
      k_(B_[0].cols()) {
  // TODO (@Brian-Acosta) validate everything is the correct size
  //  probably should use an assert to avoid checking 8N matrices in non-debug
  //  builds.
}

LCS::LCS(const MatrixXd& A, const MatrixXd& B, const MatrixXd& D,
         const VectorXd& d, const MatrixXd& E, const MatrixXd& F,
         const MatrixXd& H, const VectorXd& c, const int& N, double dt)
    : LCS(vector<MatrixXd>(N, A), vector<MatrixXd>(N, B),
          vector<MatrixXd>(N, D), vector<VectorXd>(N, d),
          vector<MatrixXd>(N, E), vector<MatrixXd>(N, F),
          vector<MatrixXd>(N, H), vector<VectorXd>(N, c), dt) {}

bool LCS::HasSameDimensionsAs(const LCS& other) const {
  return (n_ == other.n_ and m_ == other.m_ and k_ == other.k_ and
          N_ == other.N_);
}

double LCS::ScaleComplementarityDynamics() {
  DRAKE_DEMAND(D_[0].norm() > 0);
  double Dn = D_[0].norm();
  double An = A_[0].norm();
  double scale = An / Dn;
  for (size_t i = 0; i < N_; ++i) {
    D_.at(i) *= scale;
    E_.at(i) /= scale;
    c_.at(i) /= scale;
    H_.at(i) /= scale;
  }
  return scale;
}

const VectorXd LCS::Simulate(VectorXd& x_init, VectorXd& u) {
  VectorXd x_final;
  VectorXd force;
  drake::solvers::MobyLCPSolver<double> LCPSolver;
  LCPSolver.SolveLcpLemke(F_[0], E_[0] * x_init + c_[0] + H_[0] * u, &force);
  x_final = A_[0] * x_init + B_[0] * u + D_[0] * force + d_[0];
  return x_final;
}

LCS LCS::CreatePlaceholderLCS(int n_x, int n_u, int n_lambda, int N,
                              double dt) {
  MatrixXd A = MatrixXd::Ones(n_x, n_x);
  MatrixXd B = MatrixXd::Zero(n_x, n_u);
  VectorXd d = VectorXd::Zero(n_x);
  MatrixXd D = MatrixXd::Ones(n_x, n_lambda);
  MatrixXd E = MatrixXd::Zero(n_lambda, n_x);
  MatrixXd F = MatrixXd::Zero(n_lambda, n_lambda);
  MatrixXd H = MatrixXd::Zero(n_lambda, n_u);
  VectorXd c = VectorXd::Zero(n_lambda);
  LCS placeholder(A, B, D, d, E, F, H, c, N, dt);
  return placeholder;
}

}  // namespace c3
