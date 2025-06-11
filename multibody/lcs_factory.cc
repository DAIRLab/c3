#include "multibody/lcs_factory.h"

#include <iostream>

#include "multibody/geom_geom_collider.h"

#include "drake/common/text_logging.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/moby_lcp_solver.h"

using std::set;
using std::vector;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::SortedPair;
using drake::VectorX;
using drake::geometry::GeometryId;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::multibody::MultibodyForces;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace c3 {
namespace multibody {

// Linearizes the dynamics of a multibody plant into a Linear Complementarity
// System (LCS)
LCS LCSFactory::LinearizePlantToLCS(
    const MultibodyPlant<double>& plant, const Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad,
    const Context<AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const LCSOptions& options) {
  // LCS system(A, B, D, d, E, F, H, c, N, dt);
  int n_q = plant.num_positions();
  int n_v = plant.num_velocities();
  int n_x = n_q + n_v;
  int n_u = plant.num_actuators();

  int n_contacts = contact_geoms.size();  // Number of contact points
  int num_friction_directions =
      options.num_friction_directions;  // Friction directions

  ContactModel contact_model =
      ContactModelMap()[options.contact_model];  // Contact model type
  vector<double> mu = options.mu;                // Friction coefficients
  bool frictionless =
      (contact_model ==
       ContactModel::kFrictionlessSpring);  // Frictionless contacts
  float dt = options.dt;                    // Time step for discretization
  int N = options.N;  // Number of time steps in prediction horizon

  if (!frictionless) DRAKE_DEMAND(mu.size() == (size_t)n_contacts);

  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu.data(), mu.size());

  /*============== Formulate A, B and d Matrices ==================*/
  // Calculate mass matrix M(q)
  MatrixX<AutoDiffXd> M(n_v, n_v);
  plant_ad.CalcMassMatrix(context_ad, &M);

  // Calculate Coriolis term C(q, v)v
  AutoDiffVecXd C(n_v);
  plant_ad.CalcBiasTerm(context_ad, &C);

  // Calculate generalized forces τ(u) = Bu
  auto B_dyn_ad = plant_ad.MakeActuationMatrix();
  AutoDiffVecXd tau_u =
      B_dyn_ad * plant_ad.get_actuation_input_port().Eval(context_ad);

  // Calculate generalized forces due to gravity τ₍g₎
  AutoDiffVecXd tau_g = plant_ad.CalcGravityGeneralizedForces(context_ad);

  // Get forces applied to the plant
  MultibodyForces<AutoDiffXd> f_app(plant_ad);
  plant_ad.CalcForceElementsContribution(context_ad, &f_app);

  // f(q, v, u) =  M(q)⁻¹(τ(u) + τ₍g₎ + fₐₚₚ(q, v, u) - C(q, v))
  AutoDiffVecXd f_qvu =
      M.ldlt().solve(tau_g + tau_u + f_app.generalized_forces() - C);

  // f(q*, v*, u*)
  VectorXd f_qvu_norminal = ExtractValue(f_qvu);
  // Jacobian of f(q, v, u) w.r.t. q, v, u
  MatrixXd Jf = ExtractGradient(f_qvu);
  if (Jf.cols() != n_x + n_u) {
    drake::log()->error(
        "Jacobian of f(q, v, u) has unexpected number of columns: {}. "
        "Expected: {} + {} = {}",
        Jf.cols(), n_x, n_u, n_x + n_u);
  }

  VectorXd qvu_nominal(n_q + n_v + n_u);
  qvu_nominal << plant.GetPositions(context), plant.GetVelocities(context),
      plant.get_actuation_input_port().Eval(context);
  VectorXd Jf_qvu_nominal = Jf * qvu_nominal;
  // dᵥ = f(q*, v*, u*) - Jf * (q*, v*, u*)
  VectorXd d_v = f_qvu_norminal - Jf_qvu_nominal;

  // State dependent mapping q̇ = N(q)v
  Eigen::SparseMatrix<double> Nqt;
  Nqt = plant.MakeVelocityToQDotMap(context);
  MatrixXd qdotNv = MatrixXd(Nqt);

  MatrixXd Jf_q = Jf.block(0, 0, n_v, n_q);
  MatrixXd Jf_v = Jf.block(0, n_q, n_v, n_v);
  MatrixXd Jf_u = Jf.block(0, n_x, n_v, n_u);

  // Matrices for contact-free dynamics
  MatrixXd A(n_x, n_x);
  MatrixXd B(n_x, n_u);
  VectorXd d(n_x);

  // Formulate A matrix
  A.block(0, 0, n_q, n_q) =
      MatrixXd::Identity(n_q, n_q) + dt * dt * qdotNv * Jf_q;
  A.block(0, n_q, n_q, n_v) = dt * qdotNv + dt * dt * qdotNv * Jf_v;
  A.block(n_q, 0, n_v, n_q) = dt * Jf_q;
  A.block(n_q, n_q, n_v, n_v) = dt * Jf_v + MatrixXd::Identity(n_v, n_v);

  // Formulate B matrix
  B.block(0, 0, n_q, n_u) = dt * dt * qdotNv * Jf_u;
  B.block(n_q, 0, n_v, n_u) = dt * Jf_u;

  // Formulate d vector
  d.head(n_q) = dt * dt * qdotNv * d_v;
  d.tail(n_v) = dt * d_v;

  /*============== Formulate A, B and d Matrices ==================*/
  /*============== Calculate Contact Jacobians ==================*/

  // State dependent inverse mapping v = N⁺(q)⋅q̇
  Eigen::SparseMatrix<double> NqI;
  NqI = plant.MakeQDotToVelocityMap(context);
  MatrixXd vNqdot = MatrixXd(NqI);

  VectorXd phi(n_contacts);       // Signed distance values for contacts
  MatrixXd J_n(n_contacts, n_v);  // Normal contact Jacobian
  MatrixXd J_t(2 * n_contacts * num_friction_directions,
               n_v);  // Tangential contact Jacobian

  Eigen::Vector3d planar_normal = {0, 1, 0};
  double phi_i;
  MatrixX<double> J_i;
  for (int i = 0; i < n_contacts; i++) {
    multibody::GeomGeomCollider collider(plant, contact_geoms[i]);
    if (frictionless || num_friction_directions == 1)
      std::tie(phi_i, J_i) = collider.EvalPlanar(context, planar_normal);
    else
      std::tie(phi_i, J_i) =
          collider.EvalPolytope(context, num_friction_directions);

    // Signed distance value for contact i
    phi(i) = phi_i;

    // J_i is 3 x n_v
    // row (0) is contact normal
    // rows (1-num_friction directions) are the contact tangents
    J_n.row(i) = J_i.row(0);
    if (frictionless)
      continue;  // If frictionless, we only need the normal force
    J_t.block(2 * i * num_friction_directions, 0, 2 * num_friction_directions,
              n_v) = J_i.block(1, 0, 2 * num_friction_directions, n_v);
  }

  /*============== Calculate Contact Jacobians ==================*/
  /*============== Formulate D, E, F, G and c Matrices ==================*/
  int n_lambda = GetNumContactVariables(options);

  // Matrices with contact variables
  MatrixXd D = MatrixXd::Zero(n_x, n_lambda);
  MatrixXd E = MatrixXd::Zero(n_lambda, n_x);
  MatrixXd F = MatrixXd::Zero(n_lambda, n_lambda);
  MatrixXd H = MatrixXd::Zero(n_lambda, n_u);
  VectorXd c = VectorXd::Zero(n_lambda);

  if (contact_model == ContactModel::kStewartAndTrinkle) {
    FormulateStewartTrinkleContactDynamics(
        plant, context, n_q, n_v, n_u, n_contacts, num_friction_directions, dt,
        phi, J_n, J_t, Jf_q, Jf_v, Jf_u, d_v, vNqdot, qdotNv, muXd, M, D, E, F,
        H, c);
  } else if (contact_model == ContactModel::kAnitescu) {
    FormulateAnitescuContactDynamics(plant, context, n_q, n_v, n_contacts,
                                     n_lambda, num_friction_directions, dt, phi,
                                     J_n, J_t, Jf_q, Jf_v, Jf_u, d_v, vNqdot,
                                     qdotNv, muXd, M, D, E, F, H, c);
  } else if (contact_model ==
             ContactModel::kFrictionlessSpring) {  // Frictionless spring
    FormulateFrictionlessSpringContactDynamics(
        plant, context, n_q, n_v, n_contacts, dt, phi, J_n, qdotNv,
        options.spring_stiffness, M, D, E, F, H, c);
  } else {
    throw std::runtime_error("Unsupported contact model.");
  }
  /*============== Formulate D, E, F, G and c Matrices ==================*/

  return LCS(A, B, D, d, E, F, H, c, N, dt);  // Return the system;
}
void LCSFactory::FormulateFrictionlessSpringContactDynamics(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context, const int& n_q,
    const int& n_v, const int& n_contacts, const double& dt,
    const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& qdotNv,
    const double& spring_stiffness, MatrixX<AutoDiffXd>& M, MatrixXd& D,
    MatrixXd& E, MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(J_n.transpose());

  D.block(0, 0, n_q, n_contacts) = dt * dt * qdotNv * MinvJ_n_T;
  D.block(n_q, 0, n_v, n_contacts) = dt * MinvJ_n_T;

  F = MatrixXd::Identity(n_contacts, n_contacts);

  c = spring_stiffness * phi;
}

void LCSFactory::FormulateStewartTrinkleContactDynamics(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context, const int& n_q,
    const int& n_v, const int& n_u, const int& n_contacts,
    const int& num_friction_directions, const double& dt, const VectorXd& phi,
    const MatrixXd& J_n, const MatrixXd& J_t, const MatrixXd& Jf_q,
    const MatrixXd& Jf_v, const MatrixXd& Jf_u, const VectorXd& d_v,
    const MatrixXd& vNqdot, const MatrixXd& qdotNv, const VectorXd& mu,
    MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E, MatrixXd& F, MatrixXd& H,
    VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(J_n.transpose());
  MatrixXd MinvJ_t_T = M_ldlt.solve(J_t.transpose());

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t =
      MatrixXd::Zero(n_contacts, 2 * n_contacts * num_friction_directions);
  for (int i = 0; i < n_contacts; i++) {
    E_t.block(i, i * (2 * num_friction_directions), 1,
              2 * num_friction_directions) =
        MatrixXd::Ones(1, 2 * num_friction_directions);
  }

  // Formulate D matrix (state-force)
  D.block(0, 2 * n_contacts, n_q, 2 * n_contacts * num_friction_directions) =
      dt * dt * qdotNv * MinvJ_t_T;
  D.block(n_q, 2 * n_contacts, n_v, 2 * n_contacts * num_friction_directions) =
      dt * MinvJ_t_T;
  D.block(0, n_contacts, n_q, n_contacts) = dt * dt * qdotNv * MinvJ_n_T;
  D.block(n_q, n_contacts, n_v, n_contacts) = dt * MinvJ_n_T;

  // Formulate E matrix (force-state)
  E.block(n_contacts, 0, n_contacts, n_q) = dt * dt * J_n * Jf_q + J_n * vNqdot;
  E.block(2 * n_contacts, 0, 2 * n_contacts * num_friction_directions, n_q) =
      dt * J_t * Jf_q;
  E.block(n_contacts, n_q, n_contacts, n_v) = dt * J_n + dt * dt * J_n * Jf_v;
  E.block(2 * n_contacts, n_q, 2 * n_contacts * num_friction_directions, n_v) =
      J_t + dt * J_t * Jf_v;

  // Formulate F matrix (force-force)
  F.block(0, n_contacts, n_contacts, n_contacts) = mu.asDiagonal();
  F.block(0, 2 * n_contacts, n_contacts,
          2 * n_contacts * num_friction_directions) = -E_t;
  F.block(n_contacts, n_contacts, n_contacts, n_contacts) =
      dt * dt * J_n * MinvJ_n_T;
  F.block(n_contacts, 2 * n_contacts, n_contacts,
          2 * n_contacts * num_friction_directions) = dt * dt * J_n * MinvJ_t_T;
  F.block(2 * n_contacts, 0, 2 * n_contacts * num_friction_directions,
          n_contacts) = E_t.transpose();
  F.block(2 * n_contacts, n_contacts, 2 * n_contacts * num_friction_directions,
          n_contacts) = dt * J_t * MinvJ_n_T;
  F.block(2 * n_contacts, 2 * n_contacts,
          2 * n_contacts * num_friction_directions,
          2 * n_contacts * num_friction_directions) = dt * J_t * MinvJ_t_T;

  // Formulate H matrix (force-input)
  H.block(n_contacts, 0, n_contacts, n_u) = dt * dt * J_n * Jf_u;
  H.block(2 * n_contacts, 0, 2 * n_contacts * num_friction_directions, n_u) =
      dt * J_t * Jf_u;

  // Formulate c vector
  c.segment(n_contacts, n_contacts) =
      phi + dt * dt * J_n * d_v - J_n * vNqdot * plant.GetPositions(context);
  c.segment(2 * n_contacts, 2 * n_contacts * num_friction_directions) =
      J_t * dt * d_v;
}

void LCSFactory::FormulateAnitescuContactDynamics(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context, const int& n_q,
    const int& n_v, const int& n_contacts, const int& n_lambda,
    const int& num_friction_directions, const double& dt, const VectorXd& phi,
    const MatrixXd& J_n, const MatrixXd& J_t, const MatrixXd& Jf_q,
    const MatrixXd& Jf_v, const MatrixXd& Jf_u, const VectorXd& d_v,
    const MatrixXd& vNqdot, const MatrixXd& qdotNv, const VectorXd& mu,
    MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E, MatrixXd& F, MatrixXd& H,
    VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t =
      MatrixXd::Zero(n_contacts, 2 * n_contacts * num_friction_directions);
  for (int i = 0; i < n_contacts; i++) {
    E_t.block(i, i * (2 * num_friction_directions), 1,
              2 * num_friction_directions) =
        MatrixXd::Ones(1, 2 * num_friction_directions);
  }

  // Apply same friction coefficients to each friction direction for same
  // contact
  VectorXd anitescu_mu_vec = VectorXd::Zero(n_lambda);
  for (int i = 0; i < mu.rows(); i++) {
    anitescu_mu_vec.segment((2 * num_friction_directions) * i,
                            2 * num_friction_directions) =
        mu(i) * VectorXd::Ones(2 * num_friction_directions);
  }
  MatrixXd anitescu_mu_matrix = anitescu_mu_vec.asDiagonal();

  // Constructing friction bases Jc = EᵀJₙ + μJₜ
  MatrixXd J_c = E_t.transpose() * J_n + anitescu_mu_matrix * J_t;

  MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

  // Formulate D matrix (state-force)
  D.block(0, 0, n_q, n_lambda) = dt * dt * qdotNv * MinvJ_c_T;
  D.block(n_q, 0, n_v, n_lambda) = dt * MinvJ_c_T;

  // Formulate E matrix (force-state)
  E.block(0, 0, n_lambda, n_q) =
      dt * J_c * Jf_q + E_t.transpose() * J_n * vNqdot / dt;
  E.block(0, n_q, n_lambda, n_v) = J_c + dt * J_c * Jf_v;

  // Formulate F matrix (force-force)
  F = dt * J_c * MinvJ_c_T;

  // Formulate H matrix (force-input)
  H = dt * J_c * Jf_u;

  // Formulate c vector
  c = E_t.transpose() * phi / dt + dt * J_c * d_v -
      E_t.transpose() * J_n * vNqdot * plant.GetPositions(context) / dt;
}

std::pair<MatrixXd, std::vector<VectorXd>> LCSFactory::ComputeContactJacobian(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSOptions& options) {
  int n_v = plant.num_velocities();

  int n_contacts = contact_geoms.size();  // Number of contact points
  int num_friction_directions =
      options.num_friction_directions;  // Friction directions

  ContactModel contact_model =
      ContactModelMap()[options.contact_model];  // Contact model type

  bool frictionless =
      (contact_model ==
       ContactModel::kFrictionlessSpring);  // Frictionless contacts

  VectorXd phi(n_contacts);
  MatrixXd J_n(n_contacts, n_v);
  MatrixXd J_t(2 * n_contacts * num_friction_directions, n_v);
  std::vector<VectorXd> contact_points;

  for (int i = 0; i < n_contacts; i++) {
    multibody::GeomGeomCollider collider(
        plant,
        contact_geoms[i]);  // deleted num_friction_directions (check with
    // Michael about changes in geomgeom)
    auto [phi_i, J_i] = collider.EvalPolytope(context, num_friction_directions);
    auto [p_WCa, p_WCb] = collider.CalcWitnessPoints(context);
    // TODO(yangwill): think about if we want to push back both witness points
    contact_points.push_back(p_WCa);
    phi(i) = phi_i;
    J_n.row(i) = J_i.row(0);
    if (frictionless)
      continue;  // If frictionless, we only need the normal force
    J_t.block(2 * i * num_friction_directions, 0, 2 * num_friction_directions,
              n_v) = J_i.block(1, 0, 2 * num_friction_directions, n_v);
  }

  if (frictionless) {
    // if frictionless, we only need the normal jacobian
    return std::make_pair(J_n, contact_points);
  }

  if (contact_model == ContactModel::kStewartAndTrinkle) {
    // if Stewart and Trinkle model, concatenate the normal and tangential
    // jacobian
    MatrixXd J_c = MatrixXd::Zero(
        n_contacts + 2 * n_contacts * num_friction_directions, n_v);
    J_c << J_n, J_t;
    return std::make_pair(J_c, contact_points);
  }

  // Model is Anitescu
  int n_lambda = 2 * n_contacts * num_friction_directions;

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  MatrixXd E_t = MatrixXd::Zero(n_contacts, n_lambda);
  for (int i = 0; i < n_contacts; i++) {
    E_t.block(i, i * (2 * num_friction_directions), 1,
              2 * num_friction_directions) =
        MatrixXd::Ones(1, 2 * num_friction_directions);
  }

  // Apply same friction coefficients to each friction direction
  // of the same contact
  vector<double> mu = options.mu;  // Friction coefficients
  if (!frictionless) DRAKE_DEMAND(mu.size() == (size_t)n_contacts);
  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu.data(), mu.size());

  VectorXd mu_vector = VectorXd::Zero(n_lambda);
  for (int i = 0; i < muXd.rows(); i++) {
    mu_vector.segment((2 * num_friction_directions) * i,
                      2 * num_friction_directions) =
        muXd(i) * VectorXd::Ones(2 * num_friction_directions);
  }
  MatrixXd mu_matrix = mu_vector.asDiagonal();

  // Constructing friction bases  Jc = EᵀJₙ + μJₜ
  MatrixXd J_c = E_t.transpose() * J_n + mu_matrix * J_t;
  return std::make_pair(J_c, contact_points);
}

LCS LCSFactory::FixSomeModes(const LCS& other, set<int> active_lambda_inds,
                             set<int> inactive_lambda_inds) {
  vector<int> remaining_inds;

  // Assumes constant number of contacts per index
  int n_lambda = other.F()[0].rows();

  // Need to solve for lambda_active in terms of remaining elements
  // Build temporary [F1, F2] by eliminating rows for inactive
  for (int i = 0; i < n_lambda; i++) {
    // active/inactive must be exclusive
    DRAKE_ASSERT(!active_lambda_inds.count(i) ||
                 !inactive_lambda_inds.count(i));

    // In C++20, could use contains instead of count
    if (!active_lambda_inds.count(i) && !inactive_lambda_inds.count(i)) {
      remaining_inds.push_back(i);
    }
  }

  int n_remaining = remaining_inds.size();
  int n_active = active_lambda_inds.size();

  vector<MatrixXd> A, B, D, E, F, H;
  vector<VectorXd> d, c;

  // Build selection matrices:
  // S_a selects active indices
  // S_r selects remaining indices

  MatrixXd S_a = MatrixXd::Zero(n_active, n_lambda);
  MatrixXd S_r = MatrixXd::Zero(n_remaining, n_lambda);

  for (int i = 0; i < n_remaining; i++) {
    S_r(i, remaining_inds[i]) = 1;
  }
  {
    int i = 0;
    for (auto ind_j : active_lambda_inds) {
      S_a(i, ind_j) = 1;
      i++;
    }
  }

  for (int k = 0; k < other.N(); k++) {
    Eigen::BDCSVD<MatrixXd> svd;
    svd.setThreshold(1e-5);
    svd.compute(S_a * other.F()[k] * S_a.transpose(),
                Eigen::ComputeFullU | Eigen::ComputeFullV);

    // F_active likely to be low-rank due to friction, but that should be OK
    // MatrixXd res = svd.solve(F_ar);

    // Build new complementarity constraints
    // F_a_inv = pinv(S_a * F * S_a^T)
    // 0 <= \lambda_k \perp E_k x_k + F_k \lambda_k + H_k u_k + c_k
    // 0 = S_a *(E x + F S_a^T \lambda_a + F S_r^T \lambda_r + H_k u_k + c_k)
    // \lambda_a = -F_a_inv * (S_a F S_r^T * lambda_r + S_a E x + S_a H u + S_a
    // c)
    //
    // 0 <= \lambda_r \perp S_r (I - F S_a^T F_a_inv S_a) E x + ...
    //                      S_r (I - F S_a^T F_a_inv S_a) F S_r^T \lambda_r +
    //                      ... S_r (I - F S_a^T F_a_inv S_a) H u + ... S_r(I -
    //                      F S_a^T F_a_inv S_a) c
    //
    // Calling L = S_r (I - F S_a^T F_a_inv S_a)S_r * other.D()[k]
    //  E_k = L E
    //  F_k = L F S_r^t
    //  H_k = L H
    //  c_k = L c
    MatrixXd L = S_r * (MatrixXd::Identity(n_lambda, n_lambda) -
                        other.F()[k] * S_a.transpose() * svd.solve(S_a));
    MatrixXd E_k = L * other.E()[k];
    MatrixXd F_k = L * other.F()[k] * S_r.transpose();
    MatrixXd H_k = L * other.H()[k];
    MatrixXd c_k = L * other.c()[k];

    // Similarly,
    //  A_k = A - D * S_a^T * F_a_inv * S_a * E
    //  B_k = B - D * S_a^T * F_a_inv * S_a * H
    //  D_k = D * S_r^T - D * S_a^T  * F_a_inv * S_a F S_r^T
    //  d_k = d - D * S_a^T F_a_inv * S_a * c
    //
    //  Calling P = D * S_a^T * F_a_inv * S_a
    //
    //  A_k = A - P E
    //  B_k = B - P H
    //  D_k = S_r D - P S_r^T
    //  d_k = d - P c
    MatrixXd P = other.D()[k] * S_a.transpose() * svd.solve(S_a);
    MatrixXd A_k = other.A()[k] - P * other.E()[k];
    MatrixXd B_k = other.B()[k] - P * other.H()[k];
    MatrixXd D_k = other.D()[k] * S_r.transpose() - P * S_r.transpose();
    MatrixXd d_k = other.d()[k] - P * other.c()[k];
    E.push_back(E_k);
    F.push_back(F_k);
    H.push_back(H_k);
    c.push_back(c_k);
    A.push_back(A_k);
    B.push_back(B_k);
    D.push_back(D_k);
    d.push_back(d_k);
  }
  return LCS(A, B, D, d, E, F, H, c, other.dt());
}

int LCSFactory::GetNumContactVariables(ContactModel contact_model,
                                       int num_contacts,
                                       int num_friction_directions) {
  if (contact_model == ContactModel::kFrictionlessSpring) {
    return num_contacts;  // Only normal forces
  } else if (contact_model == ContactModel::kStewartAndTrinkle) {
    return 2 * num_contacts +
           2 * num_contacts *
               num_friction_directions;  // Compute contact variable count
                                         // for Stewart-Trinkle model
  } else {
    return 2 * num_contacts *
           num_friction_directions;  // Compute contact variable
                                     // count for Anitescu model
  }
}

int LCSFactory::GetNumContactVariables(const LCSOptions options) {
  multibody::ContactModel contact_model =
      ContactModelMap()[options.contact_model];
  return GetNumContactVariables(contact_model, options.num_contacts,
                                options.num_friction_directions);
}

}  // namespace multibody
}  // namespace c3