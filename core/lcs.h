#pragma once

#include <vector>
#include <Eigen/Dense>
#include "drake/common/drake_assert.h"

namespace c3 {
class LCS {
 public:
  /// Constructor for time-varying LCS
  /// @param A, B, D, d Dynamics constraints x_{k+1} = A_k x_k + B_k u_k + D_k
  /// \lambda_k + d_k
  /// @param E, F, H, c Complementarity constraints  0 <= \lambda_k \perp E_k
  /// x_k + F_k \lambda_k  + H_k u_k + c_k
  LCS(const std::vector<Eigen::MatrixXd>& A,
      const std::vector<Eigen::MatrixXd>& B,
      const std::vector<Eigen::MatrixXd>& D,
      const std::vector<Eigen::VectorXd>& d,
      const std::vector<Eigen::MatrixXd>& E,
      const std::vector<Eigen::MatrixXd>& F,
      const std::vector<Eigen::MatrixXd>& H,
      const std::vector<Eigen::VectorXd>& c, double dt);

  /// Constructor for time-invariant LCS
  LCS(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
      const Eigen::MatrixXd& D, const Eigen::VectorXd& d,
      const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& c, const int& N,
      double dt);

  LCS(const LCS& other) = default;
  LCS& operator=(const LCS&) = default;
  LCS(LCS&&) = default;
  LCS& operator=(LCS&&) = default;

  /// Simulate the system for one-step
  /// @param x_init Initial x value
  /// @param input Input value
  const Eigen::VectorXd Simulate(Eigen::VectorXd& x_init,
                                 Eigen::VectorXd& input);

 public:
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& A() const {
    return A_;
  }
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& B() const {
    return B_;
  }
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& D() const {
    return D_;
  }
  [[nodiscard]] const std::vector<Eigen::VectorXd>& d() const {
    return d_;
  }
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& E() const {
    return E_;
  }
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& F() const {
    return F_;
  }
  [[nodiscard]] const std::vector<Eigen::MatrixXd>& H() const {
    return H_;
  }
  [[nodiscard]] const std::vector<Eigen::VectorXd>& c() const {
    return c_;
  }
  [[nodiscard]] double dt() const { return dt_; }
  [[nodiscard]] int N() const { return N_; }
  [[nodiscard]] int num_states() const { return n_; }
  [[nodiscard]] int num_inputs() const { return k_; }
  [[nodiscard]] int num_lambdas() const { return m_; }

  void set_A(const std::vector<Eigen::MatrixXd>& A) {
    DRAKE_DEMAND(A.size() == N_);
    A_ = A;
  }
  void set_B(const std::vector<Eigen::MatrixXd>& B) {
    DRAKE_DEMAND(B.size() == N_);
    B_ = B;
  }
  void set_D(const std::vector<Eigen::MatrixXd>& D) {
    DRAKE_DEMAND(D.size() == N_);
    D_ = D;
  }
  void set_d(const std::vector<Eigen::VectorXd>& d) {
    DRAKE_DEMAND(d.size() == N_);
    d_ = d;
  }
  void set_E(const std::vector<Eigen::MatrixXd>& E) {
    DRAKE_DEMAND(E.size() == N_);
    E_ = E;
  }
  void set_F(const std::vector<Eigen::MatrixXd>& F) {
    DRAKE_DEMAND(F.size() == N_);
    F_ = F;
  }
  void set_H(const std::vector<Eigen::MatrixXd>& H) {
    DRAKE_DEMAND(H.size() == N_);
    H_ = H;
  }
  void set_c(const std::vector<Eigen::VectorXd>& c) {
    DRAKE_DEMAND(c.size() == N_);
    c_ = c;
  }

  void ScaleComplementarityDynamics(double scale);

  bool HasSameDimensionsAs(const LCS& other) const;

 private:
  std::vector<Eigen::MatrixXd> A_;
  std::vector<Eigen::MatrixXd> B_;
  std::vector<Eigen::MatrixXd> D_;
  std::vector<Eigen::VectorXd> d_;
  std::vector<Eigen::MatrixXd> E_;
  std::vector<Eigen::MatrixXd> F_;
  std::vector<Eigen::MatrixXd> H_;
  std::vector<Eigen::VectorXd> c_;

  size_t N_;
  double dt_;

  int n_;
  int m_;
  int k_;
};

}  // namespace c3
