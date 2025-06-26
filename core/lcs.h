#pragma once

#include <vector>
#include <Eigen/Dense>
#include "drake/common/drake_assert.h"

namespace c3 {
class LCS {
 public:

  /*!
   * Constructor for the time-varying LCS
   *        xₖ₊₁ = Aₖxₖ + Bₖuₖ + Dₖλₖ + dₖ
   *  s.t.  0 ≤ λₖ ⊥ Eₖxₖ +Fₖλₖ + Hₖuₖ + cₖ ≥ 0
   *
   *  N.B. : the dynamics matrices are already assumed to be in discrete form.
   *  dt is only used internally to C3 to properly shift the previous solution
   *  for warm-starting the optimization
   */
  LCS(const std::vector<Eigen::MatrixXd>& A,
      const std::vector<Eigen::MatrixXd>& B,
      const std::vector<Eigen::MatrixXd>& D,
      const std::vector<Eigen::VectorXd>& d,
      const std::vector<Eigen::MatrixXd>& E,
      const std::vector<Eigen::MatrixXd>& F,
      const std::vector<Eigen::MatrixXd>& H,
      const std::vector<Eigen::VectorXd>& c, double dt);

  /*!
   * Constructor for a time-invariant LCS. The same as above, but
   * the dynamics matrices are the same for each time step.
   */
  LCS(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
      const Eigen::MatrixXd& D, const Eigen::VectorXd& d,
      const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& c, const int& N,
      double dt);


  /*! Default Copy and Assignment */
  LCS(const LCS& other) = default;
  LCS& operator=(const LCS&) = default;
  LCS(LCS&&) = default;
  LCS& operator=(LCS&&) = default;

  /*! Simulate the system for one step
   * @param x_init Initial x value
   * @param u Input value
   *
   * @return The state at the next timestep
   */
  const Eigen::VectorXd Simulate(Eigen::VectorXd& x_init,
                                 Eigen::VectorXd& u);


  /*!
   * Accessors dynamics terms
   */
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

  /*!
   * Accessors for system dimensions
   */
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

  /*!
   * Scale the complementarity dynamics, such that
   * \lambda_{unscaled} = scale_factor * \lambda_{scaled}
   * result in equivalent system behavior
   */
  void ScaleComplementarityDynamics(double scale_factor);

  /*!
   * Returns true if this LCS has the same horizon and state, input, and
   * lambda dimensions as other. Otherwise returns false.
   */
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
