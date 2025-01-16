#include "qpalm_solver.h"

namespace c3::solvers {

using Eigen::VectorXd;
using Eigen::Map;


QPALMSolver::QPALMSolver(long n, long m) : data_(n, m){
  settings_.verbose = 0;
}

VectorXd QPALMSolver::Solve(const c3::solvers::QPData &qp) {
  VectorXd result = VectorXd::Zero(data_.n);
  Solve(qp, result);
  return result;
}


// TODO (@Brian-Acosta) need to convert infinite constraint bounds to
//  QPALM_INFTY?
void QPALMSolver::Solve(const c3::solvers::QPData &qp, VectorXd &x) {
  if (!init_) {
    // need to make copies to convert to the type expected by qpalm
    data_.set_Q(qp.H);
    data_.set_A(qp.A);
    data_.q = qp.g;
    data_.c = qp.c;
    data_.bmin = qp.lb;
    data_.bmax = qp.ub;
    solver_ = std::make_unique<qpalm::Solver>(data_, settings_);
    init_ = true;
  } else {
    solver_->update_Q_A(
        Map<const VectorXd>(qp.H.valuePtr(), qp.H.nonZeros()),
        Map<const VectorXd>(qp.A.valuePtr(), qp.A.nonZeros())
    );
    solver_->update_bounds(qp.lb, qp.ub);
    solver_->update_q(qp.g);
  }
  solver_->solve();
  x = solver_->get_solution().x;
}

}
