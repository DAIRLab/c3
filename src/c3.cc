#include "src/c3.h"
#include <chrono>
#include <iostream>
#include <omp.h>

#include "src/lcs.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/moby_lcp_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;
using drake::solvers::SolverOptions;

using drake::solvers::OsqpSolver;
using drake::solvers::OsqpSolverDetails;
using drake::solvers::Solve;

C3::C3(const LCS& LCS, const vector<MatrixXd>& Q, const vector<MatrixXd>& R,
       const vector<MatrixXd>& G, const vector<MatrixXd>& U,
       const vector<VectorXd>& xdesired, const C3Options& options,
       const std::vector<Eigen::VectorXd>& warm_start_delta,
       const std::vector<Eigen::VectorXd>& warm_start_binary,
       const std::vector<Eigen::VectorXd>& warm_start_x,
       const std::vector<Eigen::VectorXd>& warm_start_lambda,
       const std::vector<Eigen::VectorXd>& warm_start_u,
       bool warm_start)
    : A_(LCS.A_),
      B_(LCS.B_),
      D_(LCS.D_),
      d_(LCS.d_),
      E_(LCS.E_),
      F_(LCS.F_),
      H_(LCS.H_),
      c_(LCS.c_),
      Q_(Q),
      R_(R),
      U_(U),
      G_(G),
      xdesired_(xdesired),
      options_(options),
      N_((LCS.A_).size()),
      n_((LCS.A_)[0].cols()),
      m_((LCS.D_)[0].cols()),
      k_((LCS.B_)[0].cols()),
      hflag_(H_[0].isZero(0)),
      prog_(MathematicalProgram()),
      OSQPoptions_(SolverOptions()),
      osqp_(OsqpSolver()) {

  // Deep copy warm start
  warm_start_ = warm_start;
  if (warm_start_){
    warm_start_delta_.resize(warm_start_delta.size());
    for (size_t i = 0; i < warm_start_delta.size(); i++){
      warm_start_delta_[i] = warm_start_delta[i];
    }
    warm_start_binary_.resize(warm_start_binary.size());
    for (size_t i = 0; i < warm_start_binary.size(); i++){
      warm_start_binary_[i] = warm_start_binary[i];
    }
    warm_start_x_.resize(warm_start_x.size());
    for (size_t i = 0; i < warm_start_x.size(); i++){
      warm_start_x_[i] = warm_start_x[i];
    }
    warm_start_lambda_.resize(warm_start_lambda.size());
    for (size_t i = 0; i < warm_start_lambda.size(); i++){
      warm_start_lambda_[i] = warm_start_lambda[i];
    }
    warm_start_u_.resize(warm_start_u.size());
    for (size_t i = 0; i < warm_start_u.size(); i++){
      warm_start_u_[i] = warm_start_u[i];
    }
  }

  x_ = vector<drake::solvers::VectorXDecisionVariable>();
  u_ = vector<drake::solvers::VectorXDecisionVariable>();
  lambda_ = vector<drake::solvers::VectorXDecisionVariable>();

  for (int i = 0; i < N_ + 1; i++) {
    x_.push_back(prog_.NewContinuousVariables(n_, "x" + std::to_string(i)));
    if (i < N_) {
      u_.push_back(prog_.NewContinuousVariables(k_, "k" + std::to_string(i)));
      lambda_.push_back(
          prog_.NewContinuousVariables(m_, "lambda" + std::to_string(i)));
    }
  }

  MatrixXd LinEq(n_, 2 * n_ + k_ + m_);
  LinEq.block(0, n_ + k_ + m_, n_, n_) = -1 * MatrixXd::Identity(n_, n_);

  for (int i = 0; i < N_; i++) {
    LinEq.block(0, 0, n_, n_) = A_.at(i);
    LinEq.block(0, n_, n_, k_) = B_.at(i);
    LinEq.block(0, n_ + k_, n_, m_) = D_.at(i);

    prog_.AddLinearEqualityConstraint(
        LinEq, -d_.at(i), {x_.at(i), u_.at(i), lambda_.at(i), x_.at(i + 1)});
  }

  for (int i = 0; i < N_ + 1; i++) {
    prog_.AddQuadraticCost(Q_.at(i) * 2, -2 * Q_.at(i) * xdesired_.at(i),
                           x_.at(i), 1);
    if (i < N_) {
      prog_.AddQuadraticCost(R_.at(i) * 2, VectorXd::Zero(k_), u_.at(i), 1);
    }
  }

  OSQPoptions_.SetOption(OsqpSolver::id(), "verbose", options_.OSQP_verbose_flag);
  // OSQPoptions_.SetOption(OsqpSolver::id(), "ebs_abs", 1e-7);
  // OSQPoptions_.SetOption(OsqpSolver::id(), "eps_rel", 1e-7);
  // OSQPoptions_.SetOption(OsqpSolver::id(), "eps_prim_inf", 1e-6);
  // OSQPoptions_.SetOption(OsqpSolver::id(), "eps_dual_inf", 1e-6);
  OSQPoptions_.SetOption(OsqpSolver::id(), "max_iter",  options_.OSQP_maximum_iterations);  //30
  prog_.SetSolverOptions(OSQPoptions_);
}

VectorXd C3::Solve(VectorXd& x0, vector<VectorXd>& delta, vector<VectorXd>& w) {
  vector<MatrixXd> Gv = G_;
  VectorXd z;


  for (int i = 0; i < options_.admm_iter-1; i++) {

    z = ADMMStep(x0, &delta, &w, &Gv);

  }

  vector<VectorXd> WD(N_, VectorXd::Zero(n_ + m_ + k_));
  for (int i = 0; i < N_; i++) {
    WD.at(i) = delta.at(i) - w.at(i);
  }

  vector<VectorXd> zfin = SolveQP(x0, Gv, WD);

  z = zfin[0];

  return z.segment(n_ + m_, k_);
}

VectorXd C3::ADMMStep(VectorXd& x0, vector<VectorXd>* delta,
                      vector<VectorXd>* w, vector<MatrixXd>* Gv) {
  vector<VectorXd> WD(N_, VectorXd::Zero(n_ + m_ + k_));

  for (int i = 0; i < N_; i++) {
    WD.at(i) = delta->at(i) - w->at(i);
  }


  vector<VectorXd> z = SolveQP(x0, *Gv, WD);

  vector<VectorXd> ZW(N_, VectorXd::Zero(n_ + m_ + k_));
  for (int i = 0; i < N_; i++) {
    ZW[i] = w->at(i) + z[i];
  }


  if (U_[0].isZero(0) == 0) {
    vector<MatrixXd> Uv = U_;

    *delta = SolveProjection(Uv, ZW);
  } else {
    *delta = SolveProjection(*Gv, ZW);
  }

  for (int i = 0; i < N_; i++) {
    w->at(i) = w->at(i) + z[i] - delta->at(i);
    w->at(i) = w->at(i) / options_.rho_scale;

    Gv->at(i) = Gv->at(i) * options_.rho_scale;
  }

  return z[0];
}

vector<VectorXd> C3::SolveQP(VectorXd& x0, vector<MatrixXd>& G,
                             vector<VectorXd>& WD) {

  for (auto& constraint : constraints_) {
    prog_.RemoveConstraint(constraint);
  }
  constraints_.clear();

  constraints_.push_back(prog_.AddLinearConstraint(x_[0] == x0));

  if (hflag_ == 1) {
    drake::solvers::MobyLCPSolver<double> LCPSolver;
    VectorXd lambda0;
    LCPSolver.SolveLcpLemke(F_[0], E_[0] * x0 + c_[0], &lambda0);
    constraints_.push_back(prog_.AddLinearConstraint(lambda_[0] == lambda0));
  }

  for (auto& cost : costs_) {
    prog_.RemoveCost(cost);
  }
  costs_.clear();

  for (int i = 0; i < N_ + 1; i++) {
    if (i < N_) {
      costs_.push_back(prog_.AddQuadraticCost(
          G.at(i).block(0, 0, n_, n_) * 2,
          -2 * G.at(i).block(0, 0, n_, n_) * WD.at(i).segment(0, n_), x_.at(i),
          1));
      costs_.push_back(prog_.AddQuadraticCost(
          G.at(i).block(n_, n_, m_, m_) * 2,
          -2 * G.at(i).block(n_, n_, m_, m_) * WD.at(i).segment(n_, m_),
          lambda_.at(i), 1));
      costs_.push_back(
          prog_.AddQuadraticCost(G.at(i).block(n_ + m_, n_ + m_, k_, k_) * 2,
                                 -2 * G.at(i).block(n_ + m_, n_ + m_, k_, k_) *
                                     WD.at(i).segment(n_ + m_, k_),
                                 u_.at(i), 1));
    }
  }

  MathematicalProgramResult result = osqp_.Solve(prog_);
  VectorXd xSol = result.GetSolution(x_[0]);
  vector<VectorXd> zz(N_, VectorXd::Zero(n_ + m_ + k_));

  for (int i = 0; i < N_; i++) {
    zz.at(i).segment(0, n_) = result.GetSolution(x_[i]);
    zz.at(i).segment(n_, m_) = result.GetSolution(lambda_[i]);
    zz.at(i).segment(n_ + m_, k_) = result.GetSolution(u_[i]);

    if (warm_start_){
      // update warm start parameters
      warm_start_x_[i] = result.GetSolution(x_[i]);
      warm_start_lambda_[i] = result.GetSolution(lambda_[i]);
      warm_start_u_[i] = result.GetSolution(u_[i]);
    }
  }
  if (warm_start_)
    warm_start_x_[N_] = result.GetSolution(x_[N_]);


  return zz;
}

void C3::AddLinearConstraint(Eigen::RowVectorXd& A, double& Lowerbound,
                             double& Upperbound, int& constraint) {
  if (constraint == 1) {
    for (int i = 1; i < N_; i++) {
      userconstraints_.push_back(
          prog_.AddLinearConstraint(A, Lowerbound, Upperbound, x_.at(i)));
    }
  }

  if (constraint == 2) {
    for (int i = 0; i < N_; i++) {
      userconstraints_.push_back(
          prog_.AddLinearConstraint(A, Lowerbound, Upperbound, u_.at(i)));
    }
  }

  if (constraint == 3) {
    for (int i = 0; i < N_; i++) {
      userconstraints_.push_back(
          prog_.AddLinearConstraint(A, Lowerbound, Upperbound, lambda_.at(i)));
    }
  }
}

void C3::RemoveConstraints() {
  for (auto& userconstraint : userconstraints_) {
    prog_.RemoveConstraint(userconstraint);
  }
  userconstraints_.clear();
}

vector<VectorXd> C3::SolveProjection(vector<MatrixXd>& G,
                                     vector<VectorXd>& WZ) {
  vector<VectorXd> deltaProj(N_, VectorXd::Zero(n_ + m_ + k_));
  int i;



  if (options_.num_threads > 0) {
    omp_set_dynamic(0);  // Explicitly disable dynamic teams
    omp_set_num_threads(options_.num_threads);  // Set number of threads
  }

#pragma omp parallel for
  for (i = 0; i < N_; i++) {
    if (warm_start_){
      deltaProj[i] =
          SolveSingleProjection(G[i], WZ[i], E_[i], F_[i], H_[i], c_[i], i);
    }
    else{
      deltaProj[i] =
          SolveSingleProjection(G[i], WZ[i], E_[i], F_[i], H_[i], c_[i], -1);
    }
  }

  return deltaProj;
}

std::vector<Eigen::VectorXd> C3::GetWarmStartX() const {
  return warm_start_x_;
}

std::vector<Eigen::VectorXd> C3::GetWarmStartLambda() const {
  return warm_start_lambda_;
}

std::vector<Eigen::VectorXd> C3::GetWarmStartU() const {
  return warm_start_u_;
}

}  // namespace c3
