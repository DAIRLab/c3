#include "multibody/lcs_factory.h"

#include <iostream>

#include "multibody/multibody_utils.h"

#include "drake/common/text_logging.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/moby_lcp_solver.h"

using std::set;

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

// Helper function to expand contact_pair_configs into per-contact arrays
struct ExpandedContactConfig {
  std::vector<drake::SortedPair<drake::geometry::GeometryId>> contact_geoms;
  std::vector<int> num_friction_directions_per_contact;
  std::vector<double> mu;
  std::vector<std::optional<std::array<double, 3>>>
      planar_normal_direction_per_contact;
  int num_contacts() const { return contact_geoms.size(); }
};

ExpandedContactConfig ExpandContactPairConfigs(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context,
    const std::vector<ContactPairConfig>& configs) {
  ExpandedContactConfig result;

  for (const auto& pair : configs) {
    std::vector<GeometryId> body_A_collision_geoms =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName(pair.body_A));
    std::vector<GeometryId> body_B_collision_geoms =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName(pair.body_B));

    // Resolve indices: use all geometries if not provided or if empty
    std::vector<int> body_A_indices =
        pair.body_A_collision_geom_indices.value_or(std::vector<int>{});
    std::vector<int> body_B_indices =
        pair.body_B_collision_geom_indices.value_or(std::vector<int>{});

    if (body_A_indices.empty()) {
      // Use all geometries for body A
      body_A_indices.resize(body_A_collision_geoms.size());
      std::iota(body_A_indices.begin(), body_A_indices.end(), 0);
    } else {
      // Validate provided indices for body A
      for (int idx : body_A_indices) {
        DRAKE_DEMAND(idx >= 0 &&
                     idx < static_cast<int>(body_A_collision_geoms.size()));
      }
    }

    if (body_B_indices.empty()) {
      // Use all geometries for body B
      body_B_indices.resize(body_B_collision_geoms.size());
      std::iota(body_B_indices.begin(), body_B_indices.end(), 0);
    } else {
      // Validate provided indices for body B
      for (int idx : body_B_indices) {
        DRAKE_DEMAND(idx >= 0 &&
                     idx < static_cast<int>(body_B_collision_geoms.size()));
      }
    }

    // Create contact pairs for all combinations and replicate properties
    std::vector<drake::SortedPair<drake::geometry::GeometryId>>
        AB_geometric_pairs;
    for (int i : body_A_indices) {
      for (int j : body_B_indices) {
        AB_geometric_pairs.emplace_back(body_A_collision_geoms[i],
                                        body_B_collision_geoms[j]);
      }
    }

    // If no limit on the number of contact pairs is specified, or if the limit
    // is equal to the total number of pairs, return all pairs.
    if (pair.num_active_contact_pairs.has_value() &&
        pair.num_active_contact_pairs.value() != AB_geometric_pairs.size()) {
      AB_geometric_pairs = LCSFactory::GetNClosestContactPairs(
          plant, context, AB_geometric_pairs,
          pair.num_active_contact_pairs.value());
    }

    result.contact_geoms.insert(result.contact_geoms.end(),
                                AB_geometric_pairs.begin(),
                                AB_geometric_pairs.end());
    int n_active_contacts = AB_geometric_pairs.size();
    result.mu.insert(result.mu.end(), n_active_contacts, pair.mu);
    result.num_friction_directions_per_contact.insert(
        result.num_friction_directions_per_contact.end(), n_active_contacts,
        pair.num_friction_directions);
    result.planar_normal_direction_per_contact.insert(
        result.planar_normal_direction_per_contact.end(), n_active_contacts,
        pair.planar_normal_direction.value_or(std::array<double, 3>{}));
  }

  return result;
}

// Constructor 1: Takes only LCSFactoryOptions with contact_pair_configs
LCSFactory::LCSFactory(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    LCSFactoryOptions& options)
    : plant_(plant),
      context_(context),
      plant_ad_(plant_ad),
      context_ad_(context_ad),
      options_(options),
      contact_model_(GetContactModelMap().at(options_.contact_model)),
      n_q_(plant_.num_positions()),
      n_v_(plant_.num_velocities()),
      n_x_(n_q_ + n_v_),
      n_u_(plant_.num_actuators()),
      frictionless_(contact_model_ == ContactModel::kFrictionlessSpring),
      dt_(options.dt) {
  DRAKE_DEMAND(options.contact_pair_configs.has_value());

  // Expand contact_pair_configs into per-contact arrays using plant
  auto expanded = ExpandContactPairConfigs(
      plant, context, options.contact_pair_configs.value());

  contact_pairs_ = expanded.contact_geoms;
  n_contacts_ = contact_pairs_.size();
  mu_ = expanded.mu;

  // Create contact evaluators
  InitializeContactEvaluators(expanded.num_friction_directions_per_contact,
                              expanded.planar_normal_direction_per_contact);
}

// Constructor 2: Takes explicit contact_geoms with per-contact arrays
LCSFactory::LCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSFactoryOptions& options)
    : plant_(plant),
      context_(context),
      plant_ad_(plant_ad),
      context_ad_(context_ad),
      contact_pairs_(contact_geoms),
      options_(options),
      n_contacts_(contact_geoms.size()),
      contact_model_(GetContactModelMap().at(options_.contact_model)),
      n_q_(plant_.num_positions()),
      n_v_(plant_.num_velocities()),
      n_x_(n_q_ + n_v_),
      n_u_(plant_.num_actuators()),
      frictionless_(contact_model_ == ContactModel::kFrictionlessSpring),
      dt_(options.dt) {
  // Resolve configuration using options resolver methods
  std::vector<int> friction_dirs = options_.ResolveNumFrictionDirections();
  mu_ = options_.ResolveMu();
  std::vector<std::optional<std::array<double, 3>>> planar_normals =
      options_.ResolvePlanarNormals();

  DRAKE_DEMAND(friction_dirs.size() == (size_t)n_contacts_);
  DRAKE_DEMAND(mu_.size() == (size_t)n_contacts_);
  DRAKE_DEMAND(planar_normals.size() == (size_t)n_contacts_);

  // Create contact evaluators
  InitializeContactEvaluators(friction_dirs, planar_normals);
}

// Common initialization logic for contact evaluators
void LCSFactory::InitializeContactEvaluators(
    const std::vector<int>& friction_dirs,
    const std::vector<std::optional<std::array<double, 3>>>& planar_normals) {
  DRAKE_DEMAND(friction_dirs.size() == (size_t)n_contacts_);
  DRAKE_DEMAND(planar_normals.size() == (size_t)n_contacts_);

  contact_evaluators_.clear();
  contact_evaluators_.reserve(n_contacts_);

  for (int i = 0; i < n_contacts_; ++i) {
    if (frictionless_ || friction_dirs[i] == 1) {
      // Planar contact
      DRAKE_DEMAND(planar_normals[i].has_value());
      Eigen::Vector3d planar_normal =
          Eigen::Map<const Eigen::Vector3d>(planar_normals[i].value().data());
      contact_evaluators_.push_back(
          std::make_unique<PlanarContactEvaluator<double>>(
              plant_, contact_pairs_[i], planar_normal));
    } else {
      // Polytope contact
      contact_evaluators_.push_back(
          std::make_unique<PolytopeContactEvaluator<double>>(
              plant_, contact_pairs_[i], friction_dirs[i]));
    }
  }

  // Calculate n_lambda_ and Jt_row_sizes_ from evaluators
  std::vector<int> n_friction_directions_from_evaluators;
  for (const auto& evaluator : contact_evaluators_) {
    n_friction_directions_from_evaluators.push_back(
        evaluator->GetNumFrictionDirections());
  }

  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
      contact_model_, n_contacts_, n_friction_directions_from_evaluators);

  Jt_row_sizes_ = 2 * Eigen::Map<const VectorXi, Eigen::Unaligned>(
                          n_friction_directions_from_evaluators.data(),
                          n_friction_directions_from_evaluators.size());
}

std::vector<SortedPair<GeometryId>> LCSFactory::GetNClosestContactPairs(
    const MultibodyPlant<double>& plant, const Context<double>& context,
    const std::vector<SortedPair<GeometryId>>& contact_pairs, unsigned int N) {
  // If the limit is zero, return an empty list.
  if (N == 0) return {};

  DRAKE_DEMAND(N <= contact_pairs.size());

  // Compute signed distance for each contact pair and sort by distance. Then
  // select the N closest pairs.
  std::vector<std::pair<double, SortedPair<GeometryId>>> distance_pairs;
  distance_pairs.reserve(contact_pairs.size());

  for (const auto& geom_pair : contact_pairs) {
    multibody::GeomGeomCollider collider(plant, geom_pair);
    auto query_result = collider.GetGeometryQueryResult(context);
    distance_pairs.emplace_back(query_result.distance, geom_pair);
  }

  // Partition so that the N smallest are before distance_pairs.begin() + N
  std::nth_element(
      distance_pairs.begin(), distance_pairs.begin() + N, distance_pairs.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });

  // Extract first N elements (not sorted among themselves)
  std::vector<SortedPair<GeometryId>> active_contact_pairs;
  active_contact_pairs.reserve(N);
  for (int i = 0; i < N; ++i) {
    active_contact_pairs.push_back(distance_pairs[i].second);
  }

  return active_contact_pairs;
}

void LCSFactory::ComputeContactJacobian(VectorXd& phi, MatrixXd& Jn,
                                        MatrixXd& Jt) {
  phi.resize(n_contacts_);       // Signed distance values for contacts
  Jn.resize(n_contacts_, n_v_);  // Normal contact Jacobian
  Jt.resize(Jt_row_sizes_.sum(),
            n_v_);  // Tangential contact Jacobian

  double phi_i;
  MatrixX<double> J_i;
  for (int i = 0; i < n_contacts_; i++) {
    // Use polymorphic Eval method
    auto [phi_i, J_i] = contact_evaluators_[i]->Eval(context_);

    // Signed distance value for contact i
    phi(i) = phi_i;

    // J_i is 3 x n_v_
    // row (0) is contact normal
    // rows (1-num_friction directions) are the contact tangents
    Jn.row(i) = J_i.row(0);
    if (frictionless_)
      continue;  // If frictionless_, we only need the normal force
    Jt.block(Jt_row_sizes_.segment(0, i).sum(), 0, Jt_row_sizes_(i), n_v_) =
        J_i.block(1, 0, Jt_row_sizes_(i), n_v_);
  }
}

void LCSFactory::UpdateStateAndInput(
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  SetContext<double>(plant_, state, input, &context_);
  drake::VectorX<double> q_v_u(n_x_ + n_u_);
  q_v_u << state, input;
  drake::AutoDiffVecXd q_v_u_ad = drake::math::InitializeAutoDiff(q_v_u);
  SetPositionsAndVelocitiesIfNew<AutoDiffXd>(plant_ad_, q_v_u_ad.head(n_x_),
                                             &context_ad_);
  SetInputsIfNew<AutoDiffXd>(plant_ad_, q_v_u_ad.tail(n_u_), &context_ad_);
}
// Linearizes the dynamics of a multibody plant_ into a Linear Complementarity
// System (LCS)
LCS LCSFactory::GenerateLCS() {
  if (!frictionless_) DRAKE_DEMAND(mu_.size() == (size_t)n_contacts_);

  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu_.data(), mu_.size());

  /*============== Formulate A, B and d Matrices ==================*/
  // Calculate mass matrix M(q)
  MatrixX<AutoDiffXd> M(n_v_, n_v_);
  plant_ad_.CalcMassMatrix(context_ad_, &M);

  // Calculate Coriolis term C(q, v)v
  AutoDiffVecXd C(n_v_);
  plant_ad_.CalcBiasTerm(context_ad_, &C);

  // Calculate generalized forces τ(u) = Bu
  auto B_dyn_ad = plant_ad_.MakeActuationMatrix();
  AutoDiffVecXd tau_u =
      B_dyn_ad * plant_ad_.get_actuation_input_port().Eval(context_ad_);

  // Calculate generalized forces due to gravity τ₍g₎
  AutoDiffVecXd tau_g = plant_ad_.CalcGravityGeneralizedForces(context_ad_);

  // Get forces applied to the plant_
  MultibodyForces<AutoDiffXd> f_app(plant_ad_);
  plant_ad_.CalcForceElementsContribution(context_ad_, &f_app);

  // f(q, v, u) =  M(q)⁻¹(τ(u) + τ₍g₎ + fₐₚₚ(q, v, u) - C(q, v))
  AutoDiffVecXd f_qvu =
      M.ldlt().solve(tau_g + tau_u + f_app.generalized_forces() - C);

  // f(q*, v*, u*)
  VectorXd f_qvu_norminal = ExtractValue(f_qvu);
  // Jacobian of f(q, v, u) w.r.t. q, v, u
  MatrixXd Jf = ExtractGradient(f_qvu);
  if (Jf.cols() != n_x_ + n_u_) {
    throw std::runtime_error(fmt::format(
        "Jacobian of f(q, v, u) has unexpected number of columns: {}. "
        "Expected: {} + {} = {}",
        Jf.cols(), n_x_, n_u_, n_x_ + n_u_));
  }

  VectorXd qvu_nominal(n_q_ + n_v_ + n_u_);
  qvu_nominal << plant_.GetPositions(context_), plant_.GetVelocities(context_),
      plant_.get_actuation_input_port().Eval(context_);
  VectorXd Jf_qvu_nominal = Jf * qvu_nominal;
  // dᵥ = f(q*, v*, u*) - Jf * (q*, v*, u*)
  VectorXd d_v = f_qvu_norminal - Jf_qvu_nominal;

  // State dependent mapping q̇ = N(q)v
  Eigen::SparseMatrix<double> Nqt;
  Nqt = plant_.MakeVelocityToQDotMap(context_);
  MatrixXd qdotNv = MatrixXd(Nqt);

  MatrixXd Jf_q = Jf.block(0, 0, n_v_, n_q_);
  MatrixXd Jf_v = Jf.block(0, n_q_, n_v_, n_v_);
  MatrixXd Jf_u = Jf.block(0, n_x_, n_v_, n_u_);

  // Matrices for contact-free dynamics
  MatrixXd A(n_x_, n_x_);
  MatrixXd B(n_x_, n_u_);
  VectorXd d(n_x_);

  // Formulate A matrix
  A.block(0, 0, n_q_, n_q_) =
      MatrixXd::Identity(n_q_, n_q_) + dt_ * dt_ * qdotNv * Jf_q;
  A.block(0, n_q_, n_q_, n_v_) = dt_ * qdotNv + dt_ * dt_ * qdotNv * Jf_v;
  A.block(n_q_, 0, n_v_, n_q_) = dt_ * Jf_q;
  A.block(n_q_, n_q_, n_v_, n_v_) = dt_ * Jf_v + MatrixXd::Identity(n_v_, n_v_);

  // Formulate B matrix
  B.block(0, 0, n_q_, n_u_) = dt_ * dt_ * qdotNv * Jf_u;
  B.block(n_q_, 0, n_v_, n_u_) = dt_ * Jf_u;

  // Formulate d vector
  d.head(n_q_) = dt_ * dt_ * qdotNv * d_v;
  d.tail(n_v_) = dt_ * d_v;

  /*============== Formulate A, B and d Matrices ==================*/
  /*============== Calculate Contact Jacobians ==================*/

  // State dependent inverse mapping v = N⁺(q)⋅q̇
  Eigen::SparseMatrix<double> NqI;
  NqI = plant_.MakeQDotToVelocityMap(context_);
  MatrixXd vNqdot = MatrixXd(NqI);

  VectorXd phi;  // Signed distance values for contacts
  MatrixXd Jn;   // Normal contact Jacobian
  MatrixXd Jt;   // Tangential contact Jacobian
  ComputeContactJacobian(phi, Jn, Jt);

  /*============== Calculate Contact Jacobians ==================*/
  /*============== Formulate D, E, F, G and c Matrices ==================*/

  // Matrices with contact variables
  MatrixXd D = MatrixXd::Zero(n_x_, n_lambda_);
  MatrixXd E = MatrixXd::Zero(n_lambda_, n_x_);
  MatrixXd F = MatrixXd::Zero(n_lambda_, n_lambda_);
  MatrixXd H = MatrixXd::Zero(n_lambda_, n_u_);
  VectorXd c = VectorXd::Zero(n_lambda_);

  if (contact_model_ == ContactModel::kStewartAndTrinkle) {
    FormulateStewartTrinkleContactDynamics(phi, Jn, Jt, Jf_q, Jf_v, Jf_u, d_v,
                                           vNqdot, qdotNv, muXd, M, D, E, F, H,
                                           c);
  } else if (contact_model_ == ContactModel::kAnitescu) {
    FormulateAnitescuContactDynamics(phi, Jn, Jt, Jf_q, Jf_v, Jf_u, d_v, vNqdot,
                                     qdotNv, muXd, M, D, E, F, H, c);
  } else if (contact_model_ ==
             ContactModel::kFrictionlessSpring) {  // Frictionless spring
    FormulateFrictionlessSpringContactDynamics(
        phi, Jn, qdotNv, options_.spring_stiffness.value_or(1.0), M, D, E, F, H,
        c);
  } else {
    throw std::out_of_range("Unsupported contact model.");
  }
  /*============== Formulate D, E, F, G and c Matrices ==================*/

  return LCS(A, B, D, d, E, F, H, c, options_.N, dt_);  // Return the system;
}
void LCSFactory::FormulateFrictionlessSpringContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& qdotNv,
    const double& spring_stiffness, MatrixX<AutoDiffXd>& M, MatrixXd& D,
    MatrixXd& E, MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(Jn.transpose());

  D.block(0, 0, n_q_, n_contacts_) = dt_ * dt_ * qdotNv * MinvJ_n_T;
  D.block(n_q_, 0, n_v_, n_contacts_) = dt_ * MinvJ_n_T;

  F = MatrixXd::Identity(n_contacts_, n_contacts_);

  c = spring_stiffness * phi;
}

void LCSFactory::FormulateStewartTrinkleContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& Jt,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
    const VectorXd& mu_, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
    MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(Jn.transpose());
  MatrixXd MinvJ_t_T = M_ldlt.solve(Jt.transpose());

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t = MatrixXd::Zero(n_contacts_, Jt_row_sizes_.sum());
  for (int i = 0; i < n_contacts_; i++) {
    E_t.block(i, Jt_row_sizes_.segment(0, i).sum(), 1, Jt_row_sizes_(i)) =
        MatrixXd::Ones(1, Jt_row_sizes_(i));
  }

  // Formulate D matrix (state-force)
  D.block(0, 2 * n_contacts_, n_q_, Jt_row_sizes_.sum()) =
      dt_ * dt_ * qdotNv * MinvJ_t_T;
  D.block(n_q_, 2 * n_contacts_, n_v_, Jt_row_sizes_.sum()) = dt_ * MinvJ_t_T;
  D.block(0, n_contacts_, n_q_, n_contacts_) = dt_ * dt_ * qdotNv * MinvJ_n_T;
  D.block(n_q_, n_contacts_, n_v_, n_contacts_) = dt_ * MinvJ_n_T;

  // Formulate E matrix (force-state)
  E.block(n_contacts_, 0, n_contacts_, n_q_) =
      dt_ * dt_ * Jn * Jf_q + Jn * vNqdot;
  E.block(2 * n_contacts_, 0, Jt_row_sizes_.sum(), n_q_) = dt_ * Jt * Jf_q;
  E.block(n_contacts_, n_q_, n_contacts_, n_v_) =
      dt_ * Jn + dt_ * dt_ * Jn * Jf_v;
  E.block(2 * n_contacts_, n_q_, Jt_row_sizes_.sum(), n_v_) =
      Jt + dt_ * Jt * Jf_v;

  // Formulate F matrix (force-force)
  F.block(0, n_contacts_, n_contacts_, n_contacts_) = mu_.asDiagonal();
  F.block(0, 2 * n_contacts_, n_contacts_, Jt_row_sizes_.sum()) = -E_t;
  F.block(n_contacts_, n_contacts_, n_contacts_, n_contacts_) =
      dt_ * dt_ * Jn * MinvJ_n_T;
  F.block(n_contacts_, 2 * n_contacts_, n_contacts_, Jt_row_sizes_.sum()) =
      dt_ * dt_ * Jn * MinvJ_t_T;
  F.block(2 * n_contacts_, 0, Jt_row_sizes_.sum(), n_contacts_) =
      E_t.transpose();
  F.block(2 * n_contacts_, n_contacts_, Jt_row_sizes_.sum(), n_contacts_) =
      dt_ * Jt * MinvJ_n_T;
  F.block(2 * n_contacts_, 2 * n_contacts_, Jt_row_sizes_.sum(),
          Jt_row_sizes_.sum()) = dt_ * Jt * MinvJ_t_T;

  // Formulate H matrix (force-input)
  H.block(n_contacts_, 0, n_contacts_, n_u_) = dt_ * dt_ * Jn * Jf_u;
  H.block(2 * n_contacts_, 0, Jt_row_sizes_.sum(), n_u_) = dt_ * Jt * Jf_u;

  // Formulate c vector
  c.segment(n_contacts_, n_contacts_) =
      phi + dt_ * dt_ * Jn * d_v - Jn * vNqdot * plant_.GetPositions(context_);
  c.segment(2 * n_contacts_, Jt_row_sizes_.sum()) = Jt * dt_ * d_v;
}

void LCSFactory::FormulateAnitescuContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& Jt,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
    const VectorXd& mu_, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
    MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t = MatrixXd::Zero(n_contacts_, Jt_row_sizes_.sum());
  for (int i = 0; i < n_contacts_; i++) {
    E_t.block(i, Jt_row_sizes_.segment(0, i).sum(), 1, Jt_row_sizes_(i)) =
        MatrixXd::Ones(1, Jt_row_sizes_(i));
  }

  // Apply same friction coefficients to each friction direction for same
  // contact
  VectorXd anitescu_mu_vec = VectorXd::Zero(n_lambda_);
  for (int i = 0; i < mu_.rows(); i++) {
    anitescu_mu_vec.segment(Jt_row_sizes_.segment(0, i).sum(),
                            Jt_row_sizes_(i)) =
        mu_(i) * VectorXd::Ones(Jt_row_sizes_(i));
  }
  MatrixXd anitescu_mu_matrix = anitescu_mu_vec.asDiagonal();

  // Constructing friction bases Jc = EᵀJₙ + μJₜ
  MatrixXd J_c = E_t.transpose() * Jn + anitescu_mu_matrix * Jt;

  MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

  // Formulate D matrix (state-force)
  D.block(0, 0, n_q_, n_lambda_) = dt_ * dt_ * qdotNv * MinvJ_c_T;
  D.block(n_q_, 0, n_v_, n_lambda_) = dt_ * MinvJ_c_T;

  // Formulate E matrix (force-state)
  E.block(0, 0, n_lambda_, n_q_) =
      dt_ * J_c * Jf_q + E_t.transpose() * Jn * vNqdot / dt_;
  E.block(0, n_q_, n_lambda_, n_v_) = J_c + dt_ * J_c * Jf_v;

  // Formulate F matrix (force-force)
  F = dt_ * J_c * MinvJ_c_T;

  // Formulate H matrix (force-input)
  H = dt_ * J_c * Jf_u;

  // Formulate c vector
  c = E_t.transpose() * phi / dt_ + dt_ * J_c * d_v -
      E_t.transpose() * Jn * vNqdot * plant_.GetPositions(context_) / dt_;
}

LCS LCSFactory::LinearizePlantToLCS(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSFactoryOptions& options,
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  LCSFactory lcs_factory(plant, context, plant_ad, context_ad, contact_geoms,
                         options);
  lcs_factory.UpdateStateAndInput(state, input);
  return lcs_factory.GenerateLCS();
}

std::vector<LCSContactDescription> LCSFactory::GetContactDescriptions() {
  std::vector<LCSContactDescription> normal_contact_descriptions;
  std::vector<LCSContactDescription> tangential_contact_descriptions;

  for (int i = 0; i < n_contacts_; i++) {
    auto [p_WCa, p_WCb] = contact_evaluators_[i]->CalcWitnessPoints(context_);
    auto force_basis = contact_evaluators_[i]->CalcForceBasis(context_);

    for (int j = 0; j < force_basis.rows(); j++) {
      LCSContactDescription contact_description = {
          .witness_point_A = p_WCa,
          .witness_point_B = p_WCb,
          .force_basis = force_basis.row(j)};

      if (j == 0) {
        normal_contact_descriptions.push_back(contact_description);
      } else {
        tangential_contact_descriptions.push_back(contact_description);
      }
    }
  }

  // ===========================================================================
  // Contact Force Variable Stacking Convention
  // ===========================================================================
  //
  // The order in which contact force variables (λ) are stacked depends on the
  // contact model. This ordering is crucial for understanding the structure of
  // the LCS matrices (D, E, F, H, c).
  //
  // Stewart-Trinkle Model (n_λ = 2*n_contacts + n_tangential):
  // -----------------------------------------------------------
  // λ = [γ₁, γ₂, ..., γₙ,                    <- Slack variables (n_contacts)
  //      λₙ₁, λₙ₂, ..., λₙₙ,                 <- Normal forces (n_contacts)
  //      λₜ₁₁, λₜ₁₂, ..., λₜ₁ₘ₁,            <- Tangential forces for contact 1
  //      λₜ₂₁, λₜ₂₂, ..., λₜ₂ₘ₂,            <- Tangential forces for contact 2
  //      ...
  //      λₜₙ₁, λₜₙ₂, ..., λₜₙₘₙ]            <- Tangential forces for contact n
  //
  // where:
  //   - n = n_contacts (number of contact pairs)
  //   - mᵢ = 2 * num_friction_directions[i] (tangential force components per
  //   contact)
  //   - γᵢ are slack variables for the friction cone constraints
  //
  // Example with 2 contacts, 2 friction directions each:
  //   λ = [γ₁, γ₂, λₙ₁, λₙ₂, λₜ₁₁, λₜ₁₂, λₜ₁₃, λₜ₁₄, λₜ₂₁, λₜ₂₂, λₜ₂₃, λₜ₂₄]
  //       ↑─────↑  ↑─────↑  ↑─────────────────────↑  ↑─────────────────────↑
  //       slacks  normals  tangential (contact 1)   tangential (contact 2)
  //
  // Anitescu Model (n_λ = n_tangential):
  // -------------------------------------
  // λ = [λc₁₁, λc₁₂, ..., λc₁ₘ₁,          <- Combined forces for contact 1
  //      λc₂₁, λc₂₂, ..., λc₂ₘ₂,          <- Combined forces for contact 2
  //      ...
  //      λcₙ₁, λcₙ₂, ..., λcₙₘₙ]          <- Combined forces for contact n
  //
  // where each λcᵢⱼ represents a combined normal-tangential force:
  //   λcᵢⱼ acts along the direction: μᵢ * tangent_directionᵢⱼ +
  //   normal_directionᵢ
  //
  // Example with 2 contacts, 2 friction directions each:
  //   λ = [λc₁₁, λc₁₂, λc₁₃, λc₁₄, λc₂₁, λc₂₂, λc₂₃, λc₂₄]
  //       ↑─────────────────────↑  ↑─────────────────────↑
  //       combined (contact 1)     combined (contact 2)
  //
  // Frictionless Spring Model (n_λ = n_contacts):
  // ----------------------------------------------
  // λ = [λₙ₁, λₙ₂, ..., λₙₙ]                <- Normal forces only
  //
  // ===========================================================================

  std::vector<LCSContactDescription> contact_descriptions;
  if (contact_model_ == ContactModel::kFrictionlessSpring)
    contact_descriptions.insert(contact_descriptions.end(),
                                normal_contact_descriptions.begin(),
                                normal_contact_descriptions.end());
  else if (contact_model_ == ContactModel::kStewartAndTrinkle) {
    // Stack as: [slack variables, normal forces, tangential forces]
    for (int i = 0; i < n_contacts_; i++)
      contact_descriptions.push_back(
          LCSContactDescription::CreateSlackVariableDescription());
    contact_descriptions.insert(contact_descriptions.end(),
                                normal_contact_descriptions.begin(),
                                normal_contact_descriptions.end());
    contact_descriptions.insert(contact_descriptions.end(),
                                tangential_contact_descriptions.begin(),
                                tangential_contact_descriptions.end());
  } else if (contact_model_ == ContactModel::kAnitescu) {
    // Start with tangential basis vectors
    contact_descriptions.insert(contact_descriptions.end(),
                                tangential_contact_descriptions.begin(),
                                tangential_contact_descriptions.end());

    // For each contact, modify tangential force bases to include normal
    // component resulting in combined friction cone approximation vectors
    for (int normal_index = 0; normal_index < n_contacts_; normal_index++) {
      // Jt_row_sizes_ gives number of friction directions per contact
      for (int i = 0; i < Jt_row_sizes_(normal_index); i++) {
        int tangential_index = Jt_row_sizes_.segment(0, normal_index).sum() + i;
        DRAKE_ASSERT(
            contact_descriptions.at(tangential_index).witness_point_A ==
            normal_contact_descriptions.at(normal_index).witness_point_A);

        // Combine tangential and normal force directions:
        // force_basis = μ * tangent_direction + normal_direction
        // This creates the edges of the linearized friction cone
        contact_descriptions.at(tangential_index).force_basis =
            mu_[normal_index] *
                contact_descriptions.at(tangential_index).force_basis +
            normal_contact_descriptions.at(normal_index).force_basis;
        contact_descriptions.at(tangential_index).force_basis.normalize();
      }
    }
  }

  return contact_descriptions;
}

LCS LCSFactory::FixSomeModes(const LCS& other, set<int> active_lambda_inds,
                             set<int> inactive_lambda_inds) {
  std::vector<int> remaining_inds;

  // Assumes constant number of contacts per index
  int n_lambda_ = other.F()[0].rows();

  // Need to solve for lambda_active in terms of remaining elements
  // Build temporary [F1, F2] by eliminating rows for inactive
  for (int i = 0; i < n_lambda_; i++) {
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

  std::vector<MatrixXd> A, B, D, E, F, H;
  std::vector<VectorXd> d, c;

  // Build selection matrices:
  // S_a selects active indices
  // S_r selects remaining indices

  MatrixXd S_a = MatrixXd::Zero(n_active, n_lambda_);
  MatrixXd S_r = MatrixXd::Zero(n_remaining, n_lambda_);

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
    // \lambda_a = -F_a_inv * (S_a F S_r^T * lambda_r + S_a E x + S_a H u +
    // S_a c)
    //
    // 0 <= \lambda_r \perp S_r (I - F S_a^T F_a_inv S_a) E x + ...
    //                      S_r (I - F S_a^T F_a_inv S_a) F S_r^T \lambda_r +
    //                      ... S_r (I - F S_a^T F_a_inv S_a) H u + ... S_r(I
    //                      - F S_a^T F_a_inv S_a) c
    //
    // Calling L = S_r (I - F S_a^T F_a_inv S_a)S_r * other.D()[k]
    //  E_k = L E
    //  F_k = L F S_r^t
    //  H_k = L H
    //  c_k = L c
    MatrixXd L = S_r * (MatrixXd::Identity(n_lambda_, n_lambda_) -
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

int LCSFactory::GetNumContactVariables(
    ContactModel contact_model, int num_contacts,
    std::vector<int> num_friction_directions_per_contact) {
  if (contact_model == ContactModel::kFrictionlessSpring) {
    return num_contacts;  // Only normal forces
  } else {
    int num_tangential_lambda =
        2 * std::accumulate(num_friction_directions_per_contact.begin(),
                            num_friction_directions_per_contact.end(), 0);
    if (contact_model == ContactModel::kStewartAndTrinkle) {
      return 2 * num_contacts +
             num_tangential_lambda;  // Compute contact variable count
                                     // for Stewart-Trinkle model
    } else if (contact_model == ContactModel::kAnitescu) {
      return num_tangential_lambda;  // Compute contact variable
                                     // count for Anitescu model
    }
  }
  throw std::out_of_range("Unknown contact model.");
}

int LCSFactory::GetNumContactVariables(ContactModel contact_model,
                                       int num_contacts,
                                       int num_friction_directions) {
  std::vector<int> num_friction_directions_per_contact(num_contacts,
                                                       num_friction_directions);
  return GetNumContactVariables(contact_model, num_contacts,
                                num_friction_directions_per_contact);
}

int LCSFactory::GetNumContactVariables(
    const LCSFactoryOptions& options,
    const drake::multibody::MultibodyPlant<double>* plant) {
  multibody::ContactModel contact_model =
      GetContactModelMap().at(options.contact_model);

  int n_contacts;
  std::vector<int> n_friction_directions_per_contact;

  if (options.contact_pair_configs.has_value()) {
    // If contact pair configs are provided, they take precedence over the
    // options for number of contacts and friction directions. We can expand the
    // contact pair configs to get the actual number of contacts and friction
    // directions per contact.

    if (plant == nullptr) {
      throw std::invalid_argument(
          "plant must be provided when contact_pair_configs is set.");
    }

    // Use default context since we only need the geometry query results to
    // expand the contact pair configs, and the geometry query results do not
    // depend on the state of the plant.
    auto context = plant->CreateDefaultContext();
    auto expanded = ExpandContactPairConfigs(
        *plant, *context, options.contact_pair_configs.value());
    n_contacts = expanded.num_contacts();
    n_friction_directions_per_contact =
        expanded.num_friction_directions_per_contact;
  } else {
    n_contacts = options.ResolveNumContacts();
    n_friction_directions_per_contact = options.ResolveNumFrictionDirections();
  }

  return GetNumContactVariables(contact_model, n_contacts,
                                n_friction_directions_per_contact);
}

}  // namespace multibody
}  // namespace c3
