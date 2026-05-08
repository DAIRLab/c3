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
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using Eigen::Matrix2Xi;
using Eigen::Matrix3Xd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// TODO @bibit
//  - may need to pass LCSFactory some subset of options
// Constructor 1:
ElastoPlasticLCSFactory::ElastoPlasticLCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    ElastoPlasticLCSFactoryOptions& options)
    : LCSFactory(plant, context, plant_ad, context_ad, options) {};

// TODO @bibit
//  - may need to pass LCSFactory some subset of options
// Constructor 2:
ElastoPlasticLCSFactory::ElastoPlasticLCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& external_contact_geoms,
    const vector<SortedPair<GeometryId>>& internal_contact_geoms,
    ElastoPlasticLCSFactoryOptions& options)
    : LCSFactory(plant, context, plant_ad, context_ad, external_contact_geoms,
                 options) {};

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

// TODO @bibit
int ElastoPlasticLCSFactory::GetNumContactVariables(
    ContactModel contact_model, int num_contacts, int num_friction_directions) {
};  // Throw error
// TODO @bibit
int ElastoPlasticLCSFactory::GetNumContactVariables(
    ContactModel contact_model, int num_contacts,
    std::vector<int> num_friction_directions_per_contact) {};  // This works?
// TODO @bibit
int ElastoPlasticLCSFactory::GetNumContactVariables(
    const LCSFactoryOptions& options,
    const drake::multibody::MultibodyPlant<double>* plant) {
};  // This could work?

// TODO @bibit
LCS ElastoPlasticLCSFactory::GenerateLCS() {};

// TODO @bibit
LCS ElastoPlasticLCSFactory::LinearizePlantToLCS(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSFactoryOptions& options,
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {};

// TODO @bibit:  this is the old implementation from dairlib -- replace with new
// structure.
LCS ElastoPlasticLCSFactory::ToLCS(
    const MultibodyPlant<double>& plant, const Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad,
    const Context<AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& external_contact_geoms,
    const vector<SortedPair<GeometryId>>& internal_contact_geoms,
    const VectorXd& yield_forces, const vector<double>& mu, const double& dt,
    const int& N, int n_lambda_with_tangential,
    const vector<int>& num_friction_directions_per_contact,
    const vector<int>& starting_index_per_contact_in_lambda_t_vector,
    ContactModel contact_model) {
  // First, build the LCS without considering internal plasticity forces.
  LCS lcs_without_plasticity;
  // = LCSFactory::LinearizePlantToLCS(  // TODO @bibit had to comment out bc of
  // API change
  //   plant, context, plant_ad, context_ad, external_contact_geoms, mu, dt, N,
  //   n_lambda_with_tangential, num_friction_directions_per_contact,
  //   starting_index_per_contact_in_lambda_t_vector, contact_model);

  DRAKE_DEMAND(yield_forces.size() == internal_contact_geoms.size());

  // Dimensions.
  int n_q = plant.num_positions();
  int n_v = plant.num_velocities();
  int n_x = n_q + n_v;
  int n_u = plant.num_actuators();
  int n_lambda = lcs_without_plasticity.E()[0].rows();
  int n_external_contacts = external_contact_geoms.size();
  int n_internal_contacts = internal_contact_geoms.size();
  int n_sigma = n_internal_contacts * 2;  // Number of plasticity variables
  int n_lambda_internal = n_internal_contacts * 3;

  // Compute some relevant quantities.
  // TODO (@bibit): This has a lot of repeated computations; consider not
  // reusing LCSFactory:LinearizeToLCS and instead replicating the functionality
  // here.  This would duplicate the code instead of run-time computation.
  Eigen::SparseMatrix<double> Nqt;
  Nqt = plant.MakeVelocityToQDotMap(context);
  MatrixXd qdotNv = MatrixXd(Nqt);
  AutoDiffVecXd C(n_v);
  plant_ad.CalcBiasTerm(context_ad, &C);
  VectorXd u_dyn = plant.get_actuation_input_port().Eval(context);
  auto B_dyn_ad = plant_ad.MakeActuationMatrix();
  AutoDiffVecXd Bu =
      B_dyn_ad * plant_ad.get_actuation_input_port().Eval(context_ad);
  drake::multibody::MultibodyForces<AutoDiffXd> f_app(plant_ad);
  plant_ad.CalcForceElementsContribution(context_ad, &f_app);
  AutoDiffVecXd generalized_forces_ad;
  plant_ad.CalcGeneralizedForces(context_ad, f_app, &generalized_forces_ad);
  MatrixX<AutoDiffXd> M(n_v, n_v);
  plant_ad.CalcMassMatrix(context_ad, &M);
  AutoDiffVecXd vdot_no_contact =
      M.ldlt().solve(Bu + generalized_forces_ad - C);
  VectorXd d_vv = ExtractValue(vdot_no_contact);
  MatrixXd AB_v = ExtractGradient(vdot_no_contact);
  MatrixXd AB_v_q = AB_v.block(0, 0, n_v, n_q);
  MatrixXd AB_v_v = AB_v.block(0, n_q, n_v, n_v);
  MatrixXd AB_v_u = AB_v.block(0, n_x, n_v, n_u);
  VectorXd x_dvv(n_q + n_v + n_u);
  x_dvv << plant.GetPositions(context), plant.GetVelocities(context), u_dyn;
  VectorXd x_dvvcomp = AB_v * x_dvv;
  VectorXd d_v = d_vv - x_dvvcomp;  // = M_inv @ k
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd Ep_t = MatrixXd::Zero(n_internal_contacts, n_sigma);  // = Ep.T
  for (int i = 0; i < n_internal_contacts; i++) {
    Ep_t.block(i, 2 * i, 1, 2) = MatrixXd::Ones(1, 2);
  }
  MatrixXd J_n(n_external_contacts, n_v);       // = N.T
  MatrixXd J_t(n_lambda_with_tangential, n_v);  // = D.T
  Eigen::Vector3d planar_normal(0, 0, 1);
  for (int i = 0; i < n_external_contacts; i++) {
    multibody::GeomGeomCollider collider(plant, external_contact_geoms[i]);
    auto [phi_i, J_i] =
        (num_friction_directions_per_contact[i] == 1)
            ? collider.EvalPlanar(context, planar_normal)
            : collider.EvalPolytope(context,
                                    num_friction_directions_per_contact[i]);
    J_n.row(i) = J_i.row(0);
    J_t.block(starting_index_per_contact_in_lambda_t_vector[i], 0,
              2 * num_friction_directions_per_contact[i], n_v) =
        J_i.block(1, 0, 2 * num_friction_directions_per_contact[i], n_v);
  }
  MatrixXd Jp_t(n_sigma, n_v);  // = Dp.T
  for (int i = 0; i < n_internal_contacts; i++) {
    multibody::GeomGeomCollider collider(plant, internal_contact_geoms[i]);
    // Can collide the two node spheres together and use their normal contact
    // direction as the plasticity force direction.  The '2' is the minimum
    // number of friction directions to call EvalPolytope, but we throw out the
    // tangential directions and just consider the normal direction.
    auto [phi_i, J_i] = collider.EvalPolytope(context, 2);
    Jp_t.row(2 * i) = J_i.row(0);
    Jp_t.row(2 * i + 1) = -J_i.row(0);
  }
  MatrixXd MinvJp_t_T = M_ldlt.solve(Jp_t.transpose());  // = M_inv @ Dp
  ///////////////////////////////////////////////////////////////////////////

  /// Build them.
  /// NOTE: using complementarity variable ordering lambda_int = [slack; sigma]
  /// to match the code's S&T implementation lambda_ext = [slack; normal;
  /// tangential].
  // D
  MatrixXd D_sig = MatrixXd::Zero(n_x, n_lambda_internal);
  D_sig.block(0, n_internal_contacts, n_q, n_sigma) =
      dt * dt * qdotNv * MinvJp_t_T;
  D_sig.block(n_q, n_internal_contacts, n_v, n_sigma) = dt * MinvJp_t_T;
  // E
  MatrixXd E_sig = MatrixXd::Zero(n_lambda_internal, n_x);
  E_sig.block(n_internal_contacts, 0, n_sigma, n_q) = dt * Jp_t * AB_v_q;
  E_sig.block(n_internal_contacts, n_q, n_sigma, n_v) =
      Jp_t + dt * Jp_t * AB_v_v;
  // H
  MatrixXd H_sig = MatrixXd::Zero(n_lambda_internal, n_u);
  H_sig.block(n_internal_contacts, 0, n_sigma, n_u) = dt * Jp_t * AB_v_u;
  // F
  MatrixXd F_sig = MatrixXd::Zero(n_lambda_internal, n_lambda_internal);
  F_sig.block(0, n_internal_contacts, n_internal_contacts, n_sigma) = -Ep_t;
  F_sig.block(n_internal_contacts, 0, n_sigma, n_internal_contacts) =
      Ep_t.transpose();
  F_sig.block(n_internal_contacts, n_internal_contacts, n_sigma, n_sigma) =
      dt * Jp_t * MinvJp_t_T;
  // There are coupling terms whose structure depends on the external contact
  // model.
  MatrixXd F_sig_bl = MatrixXd::Zero(n_lambda_internal, n_lambda);
  MatrixXd F_sig_ur = MatrixXd::Zero(n_lambda, n_lambda_internal);
  if (contact_model == ContactModel::kStewartAndTrinkle) {
    // Compute a few more quantities used only by the Stewart and Trinkle
    // external contact model.
    MatrixXd MinvJ_n_T = M_ldlt.solve(J_n.transpose());  // = M_inv @ N
    MatrixXd MinvJ_t_T = M_ldlt.solve(J_t.transpose());  // = M_inv @ D

    // Now build the coupled portions of the F matrix.
    F_sig_bl.block(n_internal_contacts, n_external_contacts,
                   n_internal_contacts, n_external_contacts) =
        dt * Jp_t * MinvJ_n_T;
    F_sig_bl.block(n_internal_contacts, 2 * n_external_contacts,
                   n_internal_contacts, n_lambda_with_tangential) =
        dt * Jp_t * MinvJ_t_T;
    F_sig_ur.block(n_external_contacts, n_internal_contacts,
                   n_external_contacts, n_internal_contacts) =
        dt * dt * J_n * MinvJp_t_T;
    F_sig_ur.block(2 * n_external_contacts, n_internal_contacts,
                   n_lambda_with_tangential, n_internal_contacts) =
        dt * J_t * MinvJp_t_T;
  } else if (contact_model == ContactModel::kAnitescu) {
    // Compute a few more quantities used only by the Anitescu external contact
    // model.
    MatrixXd E_t =
        MatrixXd::Zero(n_external_contacts, n_lambda_with_tangential);
    for (int i = 0; i < n_external_contacts; i++) {
      E_t.block(i, starting_index_per_contact_in_lambda_t_vector[i], 1,
                2 * num_friction_directions_per_contact[i]) =
          MatrixXd::Ones(1, 2 * num_friction_directions_per_contact[i]);
    }
    VectorXd mu_vec = Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(
        mu.data(), mu.size());
    VectorXd anitescu_mu_vec = VectorXd::Zero(n_lambda);
    for (int i = 0; i < mu_vec.rows(); i++) {
      anitescu_mu_vec
          .segment(starting_index_per_contact_in_lambda_t_vector[i],
                   2 * num_friction_directions_per_contact[i])
          .setConstant(mu[i]);
    }
    MatrixXd anitescu_mu_matrix = anitescu_mu_vec.asDiagonal();
    MatrixXd J_c = E_t.transpose() * J_n + anitescu_mu_matrix * J_t;
    MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

    // Now build the coupled portions of the F matrix.
    F_sig_bl.block(n_internal_contacts, 0, n_sigma, n_lambda) =
        dt * Jp_t * MinvJ_c_T;
    F_sig_ur.block(0, n_internal_contacts, n_lambda, n_sigma) =
        dt * J_c * MinvJp_t_T;
  }
  // c
  VectorXd c_sig = VectorXd::Zero(n_lambda_internal);
  c_sig.segment(0, n_internal_contacts) = yield_forces;
  c_sig.segment(n_internal_contacts, n_sigma) = dt * Jp_t * d_v;

  /// Piece together the matrices from original and plasticity blocks.
  // A, B, and d are unchanged when adding internal forces.
  MatrixXd A = lcs_without_plasticity.A()[0];
  MatrixXd B = lcs_without_plasticity.B()[0];
  VectorXd d = lcs_without_plasticity.d()[0];
  // D
  MatrixXd D = MatrixXd::Zero(n_x, n_lambda + n_lambda_internal);
  D.block(0, 0, n_x, n_lambda) = lcs_without_plasticity.D()[0];
  D.block(0, n_lambda, n_x, n_lambda_internal) = D_sig;
  // E
  MatrixXd E = MatrixXd::Zero(n_lambda + n_lambda_internal, n_x);
  E.block(0, 0, n_lambda, n_x) = lcs_without_plasticity.E()[0];
  E.block(n_lambda, 0, n_lambda_internal, n_x) = E_sig;
  // F
  MatrixXd F = MatrixXd::Zero(n_lambda + n_lambda_internal,
                              n_lambda + n_lambda_internal);
  F.block(0, 0, n_lambda, n_lambda) = lcs_without_plasticity.F()[0];
  F.block(n_lambda, n_lambda, n_lambda_internal, n_lambda_internal) = F_sig;
  F.block(0, n_lambda, n_lambda, n_lambda_internal) = F_sig_ur;
  F.block(n_lambda, 0, n_lambda_internal, n_lambda) = F_sig_bl;
  // H
  MatrixXd H = MatrixXd::Zero(n_lambda + n_lambda_internal, n_u);
  H.block(0, 0, n_lambda, n_u) = lcs_without_plasticity.H()[0];
  H.block(n_lambda, 0, n_lambda_internal, n_u) = H_sig;
  // c
  VectorXd c = VectorXd::Zero(n_lambda + n_lambda_internal);
  c.segment(0, n_lambda) = lcs_without_plasticity.c()[0];
  c.segment(n_lambda, n_lambda_internal) = c_sig;

  LCS system(A, B, D, d, E, F, H, c, N, dt);
  return system;
}

// TODO @bibit
void ElastoPlasticLCSFactory::FormulateInternalPlasticContactDynamics(
    const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& J_t,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
    const VectorXd& mu, MatrixX<AutoDiffXd>& M, MatrixXd& D_int,
    MatrixXd& E_int, MatrixXd& F_int, MatrixXd& H_int, VectorXd& c_int) {
  // Build some terms.
  int n_sigma = n_internal_contacts_ * 2;
  MatrixXd Jp_t;          // TODO @bibit
  VectorXd yield_forces;  // TODO @bibit

  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJp_t_T = M_ldlt.solve(Jp_t.transpose());  // = M_inv @ Dp
  MatrixXd Ep_t = MatrixXd::Zero(n_internal_contacts_, n_sigma);  // = Ep.T
  for (int i = 0; i < n_internal_contacts_; i++) {
    Ep_t.block(i, 2 * i, 1, 2) = MatrixXd::Ones(1, 2);
  }

  /// NOTE: using complementarity variable ordering lambda_int = [slack; sigma]
  /// to match the code's S&T implementation lambda_ext = [slack; normal;
  /// tangential].

  // Formulate D_int matrix (state-plasticity) (n_x, n_lambda_internal)
  D_int.block(0, n_internal_contacts_, n_q_, n_sigma) =
      dt_ * dt_ * qdotNv * MinvJp_t_T;
  D_int.block(n_q_, n_internal_contacts_, n_v_, n_sigma) = dt_ * MinvJp_t_T;

  // Formulate E_int matrix (plasticity-state) (n_lambda_internal, n_x)
  E_int.block(n_internal_contacts_, 0, n_sigma, n_q_) = dt_ * Jp_t * Jf_q;
  E_int.block(n_internal_contacts_, n_q_, n_sigma, n_v_) =
      Jp_t + dt_ * Jp_t * Jf_v;

  // Formulate F_int matrix (plasticity-plasticity) (n_lambda_internal,
  // n_lambda_internal)
  F_int.block(0, n_internal_contacts_, n_internal_contacts_, n_sigma) = -Ep_t;
  F_int.block(n_internal_contacts_, 0, n_sigma, n_internal_contacts_) =
      Ep_t.transpose();
  F_int.block(n_internal_contacts_, n_internal_contacts_, n_sigma, n_sigma) =
      dt_ * Jp_t * MinvJp_t_T;

  // Formulate H_int matrix (plasticity-input) (n_lambda_internal, n_u)
  H_int.block(n_internal_contacts_, 0, n_sigma, n_u_) = dt_ * Jp_t * Jf_u;

  // Formulate c_int vector (n_lambda_internal)
  c_int.segment(0, n_internal_contacts_) = yield_forces;
  c_int.segment(n_internal_contacts_, n_sigma) = dt_ * Jp_t * d_v;
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
