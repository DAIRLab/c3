#pragma once

#include <set>

#include "core/lcs.h"
#include "multibody/elastoplastic_lcs_factory_options.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/query_object.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/multibody_plant.h"

using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using std::vector;

namespace c3 {
namespace multibody {

/**
 * @enum DeformationModel
 * @brief Enum representing different deformation (i.e. internal contact)
 * models.
 */
enum class DeformationModel {
  kUnknown,                ///< Unknown deformation model.
  kPlastic,                ///< Pure plastic deformation model.
  kElastic,                ///< Pure elastic deformation model.
  kSeriesElastoPlastic,    ///< Series elasto-plastic deformation model.
  kParallelElastoPlastic,  ///< Parallel elasto-plastic deformation model.
  kCompoundElastoPlastic   ///< Compound elasto-plastic deformation model (i.e.
                           ///< both series and parallel elastoplastic
                           ///< deformation components).
};

/**
 * @struct DeformationModelMap
 * @brief A map for converting string representations of deformation models to
 * their enum values.
 */
inline const std::map<std::string, DeformationModel>& GetDeformationModelMap() {
  static const std::map<std::string, DeformationModel> kDeformationModelMap = {
      {"plastic", DeformationModel::kPlastic},
      {"elastic", DeformationModel::kElastic},
      {"series_elastoplastic", DeformationModel::kSeriesElastoPlastic},
      {"parallel_elastoplastic", DeformationModel::kParallelElastoPlastic},
      {"compound_elastoplastic", DeformationModel::kCompoundElastoPlastic}};
  return kDeformationModelMap;
}

// NOTE @bibit:  can reuse LCSContactDescription, no need for elastoplastic
// extension.

class ElastoPlasticLCSFactory : LCSFactory {
 public:
  /**
   * @brief Constructor for the ElastoPlasticLCSFactory class.
   *
   * @param plant The standard MultibodyPlant templated on `double`.
   * @param context The context about which to linearize (templated on
   * `double`).
   * @param plant_ad An AutoDiffXd templated MultibodyPlant for gradient
   * calculation.
   * @param context_ad The context about which to linearize (templated on
   * `AutoDiffXd`).
   * @param options Options for elastoplastic LCS creation, including friction
   * properties, contact model, and deformation model.
   */
  ElastoPlasticLCSFactory(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      ElastoPlasticLCSFactoryOptions& options);
  /**
   * @brief Same as above, but with external and internal geometry pairs
   * specified outside of factory options.  Has the following additional input
   * arguments:
   *
   * @param external_contact_geoms Vector of geometry pairs defining external
   * contact points.
   * @param internal_contact_geoms Vector of geometry pairs defining internal
   * contact points.
   */
  ElastoPlasticLCSFactory(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          external_contact_geoms,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          internal_contact_geoms,
      ElastoPlasticLCSFactoryOptions& options);
  /**
   * TODO @bibit:  consider versions with just one of external/internal contact
   * geoms.
   *
   * TODO @bibit:  do I need to override the constructors that have
   * LCSFactoryOptions instead of ElastoPlasticLCSFactoryOptions?  At least may
   * want to override and throw an error that directs users to use the EPLCSF
   * constructor(s).
   */

  // TODO @bibit:  fairly sure these can reuse same LCSFactory implementations:
  // GetNClosestContactPairs
  // UpdateStateAndInput
  // FormulateFrictionlessSpringContactDynamics (private)
  // FormulateStewartAndTrinkleContactDynamics (private)
  // FormulateAnitescuContactDynamics (private)
  // ComputeContactJacobian (private) -- will add a separate plasticity one
  // FixSomeModes

  // TODO @bibit:  these might be able to use the same implementation

  // TODO @bibit:  determine what these need to be

  /**
   * @brief Finds the witness points for each contact pair.
   *
   * @return A pair of vectors containing the witness points on each geometry
   * for each contact pair.  This stacks the external contact descriptions
   * first, followed by the internal contact descriptions, so the first
   * n_lambda_ entries correspond to external contacts and the next
   * n_lambda_internal_ entries correspond to internal contacts.
   *
   * TODO @bibit:
   *  - store internal contact evaluators as internal_contact_evaluators_
   */
  std::vector<LCSContactDescription> GetContactDescriptions() override;

  // TODO @bibit:  probably needs a custom implementation
  // InitializeContactEvaluators (private)
  //  - Might be able to call the parent implementation for the external contact
  //    evaluators, then also initialize internal contact evaluators.
  //  - Or could add a new function InitializeInternalContactEvaluators and not
  //    override the parent implementation.

  // TODO @bibit:  probably needs a custom implementation
  // - Might throw an error if certain input argument sets are used, and require
  //   the num_friction_directions_per_contact argument.
  static int GetNumContactVariables(
      ContactModel contact_model, int num_contacts,
      int num_friction_directions);  // Throw error
  static int GetNumContactVariables(
      ContactModel contact_model, int num_contacts,
      std::vector<int> num_friction_directions_per_contact);  // This works
  static int GetNumContactVariables(
      const LCSFactoryOptions& options,
      const drake::multibody::MultibodyPlant<double>* plant =
          nullptr);  // This could work
  /**
   * @brief Get the Num Contact Variables object based on the internal state of
   * the factory.
   *
   * This method returns the number of external contact variables (n_lambda_)
   * plus internal contact variables (n_sigma_) that were computed during the
   * construction of the LCSFactory. This value is determined by the contact
   * model and the number of contacts, and is used to define the size of the
   * contact force variable in the generated LCS.
   *
   * @return int
   */
  [[nodiscard]] int GetNumContactVariables() const {
    return n_lambda_ + n_lambda_internal_;
  }

  // TODO @bibit:  probably needs a custom implementation
  LCS GenerateLCS() override;

  // TODO @bibit:  probably needs a custom implementation
  // - Might just want to throw an error if this method is called as-is and
  //   introduce a new set of input arguments that requires external and
  //   internal contact geometries separately plus
  //   ElastoPlasticLCSFactoryOptions instead of LCSFactoryOptions.
  static LCS LinearizePlantToLCS(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          contact_geoms,
      const LCSFactoryOptions& options,
      const Eigen::Ref<const drake::VectorX<double>>& state,
      const Eigen::Ref<const drake::VectorX<double>>& input);

  // TODO @bibit:  this is the old implementation from dairlib
  /// Build a time-invariant LCS that represents a system including the dynamics
  /// of an elastoplastic network linearized about a given state.  The
  /// complementarity variables include internal plasticity forces, in addition
  /// to any external contacts.
  /// NOTE:  Is compatible with LCSFactory::PreProcessor to obtain
  /// external_contact_geoms that can then be passed into this method.
  static LCS ToLCS(
      const MultibodyPlant<double>& plant, const Context<double>& context,
      const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      const Context<drake::AutoDiffXd>& context_ad,
      const vector<SortedPair<GeometryId>>& external_contact_geoms,
      const vector<SortedPair<GeometryId>>& internal_contact_geoms,
      const Eigen::VectorXd& yield_forces, const vector<double>& mu,
      const double& dt, const int& N, int n_lambda_with_tangential,
      const vector<int>& num_friction_directions_per_contact,
      const vector<int>& starting_index_per_contact_in_lambda_t_vector,
      ContactModel contact_model);

 protected:
  // TODO @bibit:  I think just the private function overrides need to go here

 private:
  /**
   * @brief Formulates the internal plasticity dynamics for a pure plastic
   * deformation model.
   *
   * @param phi Vector of signed distances.
   * @param J_n Contact Jacobian for normal forces.
   * @param J_t Contact Jacobian for tangential forces.
   * @param Jf_q Jacobian of the friction cone constraints with respect to
   * configuration.
   * @param Jf_v Jacobian of the friction cone constraints with respect to
   * velocity.
   * @param Jf_u Jacobian of the friction cone constraints with respect to
   * input.
   * @param d_v Vector of viscous friction coefficients.
   * @param vNqdot Matrix relating joint velocities to normal contact
   * velocities.
   * @param qdotNv Matrix relating joint velocities to normal contact
   * velocities.
   * @param mu Vector of friction coefficients.
   * @param[out] M Mass matrix.
   * @param[out] D_int Damping matrix.
   * @param[out] E_int Input matrix.
   * @param[out] F_int Contact force mapping matrix.
   * @param[out] H_int Complementarity constraint matrix.
   * @param[out] c_int Constant vector.
   *
   * TODO @bibit:
   *  - finish implementation
   *  - tweak input arguments as needed
   *  - determine where the F matrix coupling blocks should be computed
   */
  void FormulateInternalPlasticContactDynamics(
      const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& J_t,
      const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
      const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
      const VectorXd& mu, MatrixX<AutoDiffXd>& M, MatrixXd& D_int,
      MatrixXd& E_int, MatrixXd& F_int, MatrixXd& H_int, VectorXd& c_int);

  /**
   * @brief Computes the contact Jacobian matrix for plastic forces.
   *
   * @param[out] phi Vector of signed distances.
   * @param[out] Jp Contact Jacobian for internal plastic forces.
   *
   * TODO @bibit:
   *  - call this to define Jp
   */
  void ComputeInternalContactJacobian(VectorXd& phi, MatrixXd& Jp);

  std::vector<drake::SortedPair<drake::geometry::GeometryId>>
      internal_contact_pairs_;
  ElastoPlasticLCSFactoryOptions options_;
  int n_internal_contacts_;
  std::vector<std::unique_ptr<BidirectionalOneDimContactEvaluator<double>>>
      internal_contact_evaluators_;
  DeformationModel deformation_model_;
  int n_lambda_internal_;  // = 3 * n_internal_contacts_
  VectorXi Jp_row_sizes_;  // TODO @bibit:  I think this should always be 2
};

}  // namespace multibody
}  // namespace c3
