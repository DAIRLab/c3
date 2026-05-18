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

// NOTE:  can reuse LCSContactDescription, no need for elastoplastic extension.

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
  ElastoPlasticLCSFactory(const MultibodyPlant<double>& plant,
                          Context<double>& context,
                          const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
                          Context<drake::AutoDiffXd>& context_ad,
                          const ElastoPlasticLCSFactoryOptions& options);
  /**
   * @brief Same as above, but with external and internal geometry pairs
   * specified outside of factory options.  Has the following additional input
   * arguments:
   *
   * @param external_contact_geoms Vector of geometry pairs defining external
   * contact points.
   * @param internal_contact_geoms Vector of geometry pairs defining internal
   * contact points.
   * @param yield_forces Vector of yield forces for each internal contact point
   * (must be same length as internal_contact_geoms).
   */
  ElastoPlasticLCSFactory(
      const MultibodyPlant<double>& plant, Context<double>& context,
      const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      Context<drake::AutoDiffXd>& context_ad,
      const vector<SortedPair<GeometryId>>& external_contact_geoms,
      const vector<SortedPair<GeometryId>>& internal_contact_geoms,
      const vector<double>& yield_forces,
      const ElastoPlasticLCSFactoryOptions& options);
  /**
   * TODO @bibit:  consider versions with just one of external/internal contact
   * geoms.
   *
   * TODO @bibit:  do I need to override the constructors that have
   * LCSFactoryOptions instead of ElastoPlasticLCSFactoryOptions?  At least may
   * want to override and throw an error that directs users to use the EPLCSF
   * constructor(s).
   */

  /**
   * @brief Finds the witness points for each contact pair.
   *
   * @return A pair of vectors containing the witness points on each geometry
   * for each contact pair.  This stacks the external contact descriptions
   * first, followed by the internal contact descriptions, so the first
   * n_lambda_ entries correspond to external contacts and the next
   * n_lambda_internal_ entries correspond to internal contacts.
   */
  vector<LCSContactDescription> GetContactDescriptions() override;

  /**
   * @brief Generates a Linear Complementarity System (LCS).
   *
   * @return LCS The resulting Linear Complementarity System.
   */
  LCS GenerateLCS() override;

  /**
   * @brief Overwrites the base class's static method and throws an error since
   * internal contact geometries and yield forces are also needed to generate an
   * elastoplastic LCS.
   */
  static LCS LinearizePlantToLCS(
      const MultibodyPlant<double>& plant, Context<double>& context,
      const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      Context<drake::AutoDiffXd>& context_ad,
      const vector<SortedPair<GeometryId>>& contact_geoms,
      const LCSFactoryOptions& options,
      const Eigen::Ref<const drake::VectorX<double>>& state,
      const Eigen::Ref<const drake::VectorX<double>>& input);
  /**
   * @brief Linearizes the dynamics of a multibody plant into a Linear
   * Complementarity System (LCS) with elastoplastic internal forces.
   *
   * This method uses two copies of the Context, one for double and one for
   * AutoDiffXd, to perform gradient calculations. Contacts are specified by the
   * pairs in `contact_geoms`, where each element defines a collision.
   *
   * @param plant The standard MultibodyPlant templated on `double`.
   * @param context The context about which to linearize (templated on
   * `double`).
   * @param plant_ad An AutoDiffXd templated MultibodyPlant for gradient
   * calculation.
   * @param context_ad The context about which to linearize (templated on
   * `AutoDiffXd`).
   * @param external_contact_geoms Vector of geometry pairs defining external
   * contact points.
   * @param internal_contact_geoms Vector of geometry pairs defining internal
   * contact points.
   * @param yield_forces Vector of yield forces for each internal contact point
   * (must be same length as internal_contact_geoms).
   * @param options Options for LCS creation, including friction properties and
   * contact model.
   * @param state The state vector at which to linearize.
   * @param input The input vector at which to linearize.
   * @return LCS The resulting Linear Complementarity System.
   */
  static LCS LinearizePlantToLCS(
      const MultibodyPlant<double>& plant, Context<double>& context,
      const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      Context<drake::AutoDiffXd>& context_ad,
      const vector<SortedPair<GeometryId>>& external_contact_geoms,
      const vector<SortedPair<GeometryId>>& internal_contact_geoms,
      const vector<double>& yield_forces,
      const ElastoPlasticLCSFactoryOptions& options,
      const Eigen::Ref<const drake::VectorX<double>>& state,
      const Eigen::Ref<const drake::VectorX<double>>& input);

 private:
  /**
   * @brief Initializes contact evaluators for all internal contact pairs.
   *
   * This method creates and configures BidirectionalOneDimContactEvaluator
   * objects for each contact pair, setting up the friction directions as
   * specified.  No input arguments are needed; relies only on the class
   * variable n_internal_contacts_.
   */
  void InitializeInternalContactEvaluators();

  /**
   * @brief Formulates the internal plasticity dynamics for a pure plastic
   * deformation model.
   *
   * @param Jn Contact Jacobian for normal forces.
   * @param Jt Contact Jacobian for tangential forces.
   * @param Jp Contact Jacobian for internal plastic forces.
   * @param Jf_q Jacobian of the friction cone constraints with respect to
   * configuration.
   * @param Jf_v Jacobian of the friction cone constraints with respect to
   * velocity.
   * @param Jf_u Jacobian of the friction cone constraints with respect to
   * input.
   * @param d_v Vector of viscous friction coefficients.
   * @param qdotNv Matrix relating joint velocities to normal contact
   * velocities.
   * @param mu Vector of friction coefficients.
   * @param[out] M Mass matrix.
   * @param[out] D_int Damping matrix.
   * @param[out] E_int Input matrix.
   * @param[out] F_int Contact force mapping matrix.
   * @param[out] H_int Complementarity constraint matrix.
   * @param[out] c_int Constant vector.
   * @param[out] F_coupling_bl Bottom left matrix for external-internal force
   * coupling.
   * @param[out] F_coupling_ur Upper right matrix for external-internal force
   * coupling.
   */
  void FormulateInternalPlasticContactDynamics(
      const MatrixXd& Jn, const MatrixXd& Jt, const MatrixXd& Jp,
      const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
      const VectorXd& d_v, const MatrixXd& qdotNv, const VectorXd& mu,
      MatrixX<AutoDiffXd>& M, MatrixXd& D_int, MatrixXd& E_int, MatrixXd& F_int,
      MatrixXd& H_int, VectorXd& c_int, MatrixXd& F_coupling_bl,
      MatrixXd& F_coupling_ur);

  /**
   * @brief Computes the contact Jacobian matrix for plastic forces.
   *
   * @param[out] phi Vector of signed distances for internal plastic forces.
   * These are not to be used anywhere unless there are parallel springs in the
   * deformation model, which is yet to be implemented.
   * @param[out] Jp Contact Jacobian for internal plastic forces.
   */
  void ComputeInternalContactJacobian(VectorXd& phi, MatrixXd& Jp);

  vector<SortedPair<GeometryId>> internal_contact_pairs_;
  ElastoPlasticLCSFactoryOptions options_;
  int n_internal_contacts_;
  int n_lambda_internal_;  // = 3 * n_internal_contacts_
  vector<std::unique_ptr<BidirectionalOneDimContactEvaluator<double>>>
      internal_contact_evaluators_;
  DeformationModel deformation_model_;
  vector<double> yield_forces_;
};

}  // namespace multibody
}  // namespace c3
