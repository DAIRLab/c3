#include "multibody/elastoplastic_lcs_factory.h"

namespace c3 {
namespace multibody {

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::multibody::MultibodyForces;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using Eigen::Matrix2Xi;
using Eigen::Matrix3Xd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Helper function to expand internal_contact_pair_configs into per-contact
// arrays
// TODO @bibit:  If other deformation models are implemented besides plastic,
// this needs to include additional parameters.
struct ExpandedInternalContactConfig {
  vector<SortedPair<GeometryId>> contact_geoms;
  vector<double> yield_forces;
  int num_internal_contacts() const { return contact_geoms.size(); }
};

// TODO @bibit:  If other deformation models are implemented besides plastic,
// this needs to include additional parameters.
ExpandedInternalContactConfig ExpandInternalContactPairConfigs(
    const MultibodyPlant<double>& plant, const Context<double>& context,
    const vector<ElastoPlasticContactPairConfig>& configs) {
  ExpandedInternalContactConfig result;

  for (const auto& pair : configs) {
    vector<GeometryId> body_A_collision_geoms =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName(pair.body_A));
    vector<GeometryId> body_B_collision_geoms =
        plant.GetCollisionGeometriesForBody(plant.GetBodyByName(pair.body_B));

    DRAKE_DEMAND(body_A_collision_geoms.size() == 1);
    DRAKE_DEMAND(body_B_collision_geoms.size() == 1);

    GeometryId body_A_collision_geom = body_A_collision_geoms[0];
    GeometryId body_B_collision_geom = body_B_collision_geoms[0];

    // Create a contact pair.
    SortedPair<GeometryId> contact_pair(body_A_collision_geom,
                                        body_B_collision_geom);
    result.contact_geoms.push_back(contact_pair);

    // Store other parameters.
    DRAKE_DEMAND(pair.yield_force.has_value());
    result.yield_forces.push_back(pair.yield_force.value());
  }

  return result;
}

// Constructor 1:  Takes only ElastoPlasticLCSFactoryOptions with internal
// contact pair configs.  Also assumes (external) contact pair configs are set
// the same way.
// TODO @bibit:
//  - unsure if passing ElastoPlasticLCSFactoryOptions to LCSFactory is ok.
ElastoPlasticLCSFactory::ElastoPlasticLCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    const ElastoPlasticLCSFactoryOptions& options)
    : LCSFactory(plant, context, plant_ad, context_ad, options),
      options_(options),
      deformation_model_(
          GetDeformationModelMapFromString().at(options_.deformation_model)) {
  DRAKE_DEMAND(options_.internal_contact_pair_configs.has_value());

  // Expand contact_pair_configs into per-contact arrays using plant
  auto expanded = ExpandInternalContactPairConfigs(
      plant, context, options_.internal_contact_pair_configs.value());

  internal_contact_pairs_ = expanded.contact_geoms;
  n_internal_contacts_ = internal_contact_pairs_.size();
  yield_forces_ = expanded.yield_forces;

  // Create internal contact evaluators
  InitializeInternalContactEvaluators();
};

// Constructor 2:  Takes explicit external and internal contact_geoms with per-
// contact arrays.  Also requires yield_forces for each internal contact.
// TODO @bibit:
//  - unsure if passing ElastoPlasticLCSFactoryOptions to LCSFactory is ok.
ElastoPlasticLCSFactory::ElastoPlasticLCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& external_contact_geoms,
    const vector<SortedPair<GeometryId>>& internal_contact_geoms,
    const vector<double>& yield_forces,
    const ElastoPlasticLCSFactoryOptions& options)
    : LCSFactory(plant, context, plant_ad, context_ad, external_contact_geoms,
                 options),
      internal_contact_pairs_(internal_contact_geoms),
      options_(options),
      n_internal_contacts_(internal_contact_geoms.size()),
      deformation_model_(
          GetDeformationModelMapFromString().at(options_.deformation_model)),
      yield_forces_(yield_forces) {
  DRAKE_DEMAND(yield_forces_.size() == internal_contact_pairs_.size());

  // Create internal contact evaluators
  InitializeInternalContactEvaluators();
};

vector<LCSContactDescription>
ElastoPlasticLCSFactory::GetContactDescriptions() {
  vector<LCSContactDescription> external_contact_descriptions =
      LCSFactory::GetContactDescriptions();

  // Build internal contact descriptions.
  std::vector<LCSContactDescription> internal_contact_descriptions;

  // Stack as: [slack variables, tangential forces]
  for (int i = 0; i < n_internal_contacts_; i++)
    internal_contact_descriptions.push_back(
        LCSContactDescription::CreateSlackVariableDescription());

  for (int i = 0; i < n_internal_contacts_; i++) {
    auto [p_WCa, p_WCb] =
        internal_contact_evaluators_[i]->CalcWitnessPoints(context_);
    auto force_basis =
        internal_contact_evaluators_[i]->CalcForceBasis(context_);

    for (int j = 0; j < force_basis.rows(); j++) {
      LCSContactDescription contact_description = {
          .witness_point_A = p_WCa,
          .witness_point_B = p_WCb,
          .force_basis = force_basis.row(j)};
      internal_contact_descriptions.push_back(contact_description);
    }
  }

  // Put internal contact descriptions after external.
  std::vector<LCSContactDescription> contact_descriptions;
  contact_descriptions.insert(contact_descriptions.end(),
                              external_contact_descriptions.begin(),
                              external_contact_descriptions.end());
  contact_descriptions.insert(contact_descriptions.end(),
                              internal_contact_descriptions.begin(),
                              internal_contact_descriptions.end());
  return contact_descriptions;
};

LCS ElastoPlasticLCSFactory::GenerateLCS() {
  // First generate the external-only LCS.
  LCS external_lcs = LCSFactory::GenerateLCS();

  /*============== Formulate A, B and d Matrices ==================*/

  // The A, B, and d matrices are the same as the external-only LCS.
  MatrixXd A = external_lcs.A()[0];
  MatrixXd B = external_lcs.B()[0];
  VectorXd d = external_lcs.d()[0];
  DRAKE_DEMAND(A.rows() == n_x_);
  DRAKE_DEMAND(A.cols() == n_x_);
  DRAKE_DEMAND(B.rows() == n_x_);
  DRAKE_DEMAND(B.cols() == n_u_);
  DRAKE_DEMAND(d.rows() == n_x_);

  /*============== Formulate A, B and d Matrices ==================*/
  /*============ Formulate external D, E, F, G and c Matrices ================*/

  // The D, E, F, H, and c matrices need to be augmented from the external-only
  // LCS with internal and coupling terms.
  MatrixXd D_ext = external_lcs.D()[0];
  MatrixXd E_ext = external_lcs.E()[0];
  MatrixXd F_ext = external_lcs.F()[0];
  MatrixXd H_ext = external_lcs.H()[0];
  VectorXd c_ext = external_lcs.c()[0];
  DRAKE_DEMAND(D_ext.rows() == n_x_);
  DRAKE_DEMAND(D_ext.cols() == n_lambda_);
  DRAKE_DEMAND(E_ext.rows() == n_lambda_);
  DRAKE_DEMAND(E_ext.cols() == n_x_);
  DRAKE_DEMAND(F_ext.rows() == n_lambda_);
  DRAKE_DEMAND(F_ext.cols() == n_lambda_);
  DRAKE_DEMAND(H_ext.rows() == n_lambda_);
  DRAKE_DEMAND(H_ext.cols() == n_u_);
  DRAKE_DEMAND(c_ext.size() == n_lambda_);

  /*============ Formulate external D, E, F, G and c Matrices ================*/
  /*========== Duplicate computation from LCSFactory:GenerateLCS =============*/

  // TODO @bibit:  This duplicates some code and computation; could consider a
  // more efficient way of doing this.
  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu_.data(), mu_.size());

  // State dependent inverse mapping v = N⁺(q)⋅q̇
  Eigen::SparseMatrix<double> NqI;
  NqI = plant_.MakeQDotToVelocityMap(context_);
  MatrixXd vNqdot = MatrixXd(NqI);

  VectorXd phi;  // Signed distance values for contacts
  MatrixXd Jn;   // Normal contact Jacobian
  MatrixXd Jt;   // Tangential contact Jacobian
  ComputeContactJacobian(phi, Jn, Jt);

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

  MatrixXd Jf_q = Jf.block(0, 0, n_v_, n_q_);
  MatrixXd Jf_v = Jf.block(0, n_q_, n_v_, n_v_);
  MatrixXd Jf_u = Jf.block(0, n_x_, n_v_, n_u_);

  // State dependent mapping q̇ = N(q)v
  Eigen::SparseMatrix<double> Nqt;
  Nqt = plant_.MakeVelocityToQDotMap(context_);
  MatrixXd qdotNv = MatrixXd(Nqt);

  /*========== Duplicate computation from LCSFactory:GenerateLCS =============*/
  /*============ Formulate internal D, E, F, G and c Matrices ================*/

  // Placeholders of proper size.
  MatrixXd D_int = MatrixXd::Zero(n_x_, n_lambda_internal_);
  MatrixXd E_int = MatrixXd::Zero(n_lambda_internal_, n_x_);
  MatrixXd H_int = MatrixXd::Zero(n_lambda_internal_, n_u_);
  MatrixXd F_int = MatrixXd::Zero(n_lambda_internal_, n_lambda_internal_);
  VectorXd c_int = VectorXd::Zero(n_lambda_internal_);
  MatrixXd F_coupling_bl = MatrixXd::Zero(n_lambda_internal_, n_lambda_);
  MatrixXd F_coupling_ur = MatrixXd::Zero(n_lambda_, n_lambda_internal_);

  // One remaining quantity to compute for internal contacts specifically.
  VectorXd
      phi_internal;  // Signed distance values for internal contacts (unused)
  MatrixXd Jp;       // Internal contact Jacobian
  ComputeInternalContactJacobian(phi_internal, Jp);

  // Fill the placeholders in with values.
  FormulateInternalPlasticContactDynamics(
      Jn, Jt, Jp, Jf_q, Jf_v, Jf_u, d_v, qdotNv, muXd, M, D_int, E_int, F_int,
      H_int, c_int, F_coupling_bl, F_coupling_ur);

  /*============ Formulate internal D, E, F, G and c Matrices ================*/
  /*============== Formulate full D, E, F, G and c Matrices ==================*/

  // Piece together the external and internal components.
  MatrixXd D = MatrixXd::Zero(n_x_, n_lambda_ + n_lambda_internal_);
  MatrixXd E = MatrixXd::Zero(n_lambda_ + n_lambda_internal_, n_x_);
  MatrixXd F = MatrixXd::Zero(n_lambda_ + n_lambda_internal_,
                              n_lambda_ + n_lambda_internal_);
  MatrixXd H = MatrixXd::Zero(n_lambda_ + n_lambda_internal_, n_u_);
  VectorXd c = VectorXd::Zero(n_lambda_ + n_lambda_internal_);

  // D
  D.block(0, 0, n_x_, n_lambda_) = D_ext;
  D.block(0, n_lambda_, n_x_, n_lambda_internal_) = D_int;
  // E
  E.block(0, 0, n_lambda_, n_x_) = E_ext;
  E.block(n_lambda_, 0, n_lambda_internal_, n_x_) = E_int;
  // F
  F.block(0, 0, n_lambda_, n_lambda_) = F_ext;
  F.block(n_lambda_, n_lambda_, n_lambda_internal_, n_lambda_internal_) = F_int;
  F.block(0, n_lambda_, n_lambda_, n_lambda_internal_) = F_coupling_ur;
  F.block(n_lambda_, 0, n_lambda_internal_, n_lambda_) = F_coupling_bl;
  // H
  H.block(0, 0, n_lambda_, n_u_) = H_ext;
  H.block(n_lambda_, 0, n_lambda_internal_, n_u_) = H_int;
  // c
  c.segment(0, n_lambda_) = c_ext;
  c.segment(n_lambda_, n_lambda_internal_) = c_int;

  /*============== Formulate full D, E, F, G and c Matrices ==================*/

  return LCS(A, B, D, d, E, F, H, c, options_.N, dt_);  // Return the system;
};

LCS ElastoPlasticLCSFactory::LinearizePlantToLCS(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    Context<drake::AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const LCSFactoryOptions& options,
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  throw std::runtime_error(
      "For ElastoPlasticLCSFactory, must call LinearizePlantToLCS with "
      "additional input arguments, e.g. internal_contact_geoms.");
};

LCS ElastoPlasticLCSFactory::LinearizePlantToLCS(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    Context<drake::AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& external_contact_geoms,
    const vector<SortedPair<GeometryId>>& internal_contact_geoms,
    const vector<double>& yield_forces,
    const ElastoPlasticLCSFactoryOptions& options,
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  ElastoPlasticLCSFactory lcs_factory(
      plant, context, plant_ad, context_ad, external_contact_geoms,
      internal_contact_geoms, yield_forces, options);
  lcs_factory.UpdateStateAndInput(state, input);
  return lcs_factory.GenerateLCS();
};

void ElastoPlasticLCSFactory::InitializeInternalContactEvaluators() {
  internal_contact_evaluators_.clear();
  internal_contact_evaluators_.reserve(n_internal_contacts_);

  // All internal contacts are bi-directional 1D contacts.
  for (int i = 0; i < n_internal_contacts_; i++) {
    internal_contact_evaluators_.push_back(
        std::make_unique<BidirectionalOneDimContactEvaluator<double>>(
            plant_, internal_contact_pairs_[i]));
  }

  // Internal complementarity variables are always 3 * the number of internal
  // contacts because we assume 1 slack variable + 2 plastic force directions.
  n_lambda_internal_ = 3 * n_internal_contacts_;
};

void ElastoPlasticLCSFactory::FormulateInternalPlasticContactDynamics(
    const MatrixXd& Jn, const MatrixXd& Jt, const MatrixXd& Jp,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& qdotNv, const VectorXd& mu,
    MatrixX<AutoDiffXd>& M, MatrixXd& D_int, MatrixXd& E_int, MatrixXd& F_int,
    MatrixXd& H_int, VectorXd& c_int, MatrixXd& F_coupling_bl,
    MatrixXd& F_coupling_ur) {
  // Build some terms.
  int n_sigma = n_internal_contacts_ * 2;
  int n_external_contacts = n_contacts_;

  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJp_t_T = M_ldlt.solve(Jp.transpose());  // = M_inv @ Dp
  MatrixXd Ep_t = MatrixXd::Zero(n_internal_contacts_, n_sigma);  // = Ep.T
  for (int i = 0; i < n_internal_contacts_; i++) {
    Ep_t.block(i, 2 * i, 1, 2) = MatrixXd::Ones(1, 2);
  }

  // NOTE: Use complementarity variable ordering lambda_int = [slack; sigma] to
  // match the S&T implementation lambda_ext = [slack; normal; tangential].

  // Formulate D_int matrix (state-plasticity) (n_x, n_lambda_internal)
  D_int.block(0, n_internal_contacts_, n_q_, n_sigma) =
      dt_ * dt_ * qdotNv * MinvJp_t_T;
  D_int.block(n_q_, n_internal_contacts_, n_v_, n_sigma) = dt_ * MinvJp_t_T;

  // Formulate E_int matrix (plasticity-state) (n_lambda_internal, n_x)
  E_int.block(n_internal_contacts_, 0, n_sigma, n_q_) = dt_ * Jp * Jf_q;
  E_int.block(n_internal_contacts_, n_q_, n_sigma, n_v_) = Jp + dt_ * Jp * Jf_v;

  // Formulate F_int matrix (plasticity-plasticity) (n_lambda_internal,
  // n_lambda_internal)
  F_int.block(0, n_internal_contacts_, n_internal_contacts_, n_sigma) = -Ep_t;
  F_int.block(n_internal_contacts_, 0, n_sigma, n_internal_contacts_) =
      Ep_t.transpose();
  F_int.block(n_internal_contacts_, n_internal_contacts_, n_sigma, n_sigma) =
      dt_ * Jp * MinvJp_t_T;

  // Formulate H_int matrix (plasticity-input) (n_lambda_internal, n_u)
  H_int.block(n_internal_contacts_, 0, n_sigma, n_u_) = dt_ * Jp * Jf_u;

  // Formulate c_int vector (n_lambda_internal)
  c_int.segment(0, n_internal_contacts_) =
      Eigen::Map<const VectorXd>(yield_forces_.data(), n_internal_contacts_);
  c_int.segment(n_internal_contacts_, n_sigma) = dt_ * Jp * d_v;

  // There are external-internal contact force coupling terms whose structure
  // depends on the external contact model.
  if (contact_model_ == ContactModel::kStewartAndTrinkle) {
    // Compute a few more quantities used only by the Stewart and Trinkle
    // external contact model.
    MatrixXd MinvJ_n_T = M_ldlt.solve(Jn.transpose());  // = M_inv @ N
    MatrixXd MinvJ_t_T = M_ldlt.solve(Jt.transpose());  // = M_inv @ D

    // Now build the coupled portions of the F matrix.
    F_coupling_bl.block(n_internal_contacts_, n_external_contacts, n_sigma,
                        n_external_contacts) =
        dt_ * Jp * MinvJ_n_T;
    F_coupling_bl.block(n_internal_contacts_, 2 * n_external_contacts,
                        n_sigma, Jt_row_sizes_.sum()) =
        dt_ * Jp * MinvJ_t_T;
    F_coupling_ur.block(n_external_contacts, n_internal_contacts_,
                        n_external_contacts, n_sigma) =
        dt_ * dt_ * Jn * MinvJp_t_T;
    F_coupling_ur.block(2 * n_external_contacts, n_internal_contacts_,
                        Jt_row_sizes_.sum(), n_sigma) =
        dt_ * Jt * MinvJp_t_T;
  } else if (contact_model_ == ContactModel::kAnitescu) {
    // Compute a few more quantities used only by the Anitescu external contact
    // model.
    MatrixXd E_t = MatrixXd::Zero(n_contacts_, Jt_row_sizes_.sum());
    for (int i = 0; i < n_contacts_; i++) {
      E_t.block(i, Jt_row_sizes_.segment(0, i).sum(), 1, Jt_row_sizes_(i)) =
          MatrixXd::Ones(1, Jt_row_sizes_(i));
    }
    VectorXd anitescu_mu_vec = VectorXd::Zero(n_lambda_);
    for (int i = 0; i < mu.rows(); i++) {
      anitescu_mu_vec.segment(Jt_row_sizes_.segment(0, i).sum(),
                              Jt_row_sizes_(i)) =
          mu(i) * VectorXd::Ones(Jt_row_sizes_(i));
    }
    MatrixXd anitescu_mu_matrix = anitescu_mu_vec.asDiagonal();
    MatrixXd J_c = E_t.transpose() * Jn + anitescu_mu_matrix * Jt;
    MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

    // Now build the coupled portions of the F matrix.
    F_coupling_bl.block(n_internal_contacts_, 0, n_sigma, n_lambda_) =
        dt_ * Jp * MinvJ_c_T;
    F_coupling_ur.block(0, n_internal_contacts_, n_lambda_, n_sigma) =
        dt_ * J_c * MinvJp_t_T;
  }
}

void ElastoPlasticLCSFactory::ComputeInternalContactJacobian(VectorXd& phi,
                                                             MatrixXd& Jp) {
  phi.resize(n_internal_contacts_);  // Signed distance values for contacts
  Jp.resize(2 * n_internal_contacts_, n_v_);  // Internal contact Jacobian

  double phi_i;
  MatrixX<double> J_i;
  for (int i = 0; i < n_internal_contacts_; i++) {
    // Use polymorphic Eval method
    auto [phi_i, J_i] = internal_contact_evaluators_[i]->Eval(context_);

    // Signed distance value for contact i
    phi(i) = phi_i;

    // J_i is 2 x n_v_
    // rows (0-1) are the two "plasticity force" bases in opposing directions
    Jp.block(2 * i, 0, 2, n_v_) = J_i;
  }
}

}  // namespace multibody
}  // namespace c3
