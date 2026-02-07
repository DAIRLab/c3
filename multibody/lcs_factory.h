#pragma once

#include <map>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include "core/lcs.h"
#include "multibody/lcs_factory_options.h"

#include "drake/common/sorted_pair.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

using drake::AutoDiffXd;
using drake::MatrixX;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

namespace c3 {
namespace multibody {

/**
 * @enum ContactModel
 * @brief Enum representing different contact models.
 */
enum class ContactModel {
  kUnknown,            ///< Unknown contact model.
  kStewartAndTrinkle,  ///< Stewart and Trinkle timestepping contact model.
  kAnitescu,           ///< Anitescu convex contact model.
  kFrictionlessSpring  ///< Frictionless spring contact model.
};

/**
 * @struct ContactModelMap
 * @brief A map for converting string representations of contact models to their
 * enum values.
 */
inline const std::map<std::string, ContactModel>& GetContactModelMap() {
  static const std::map<std::string, ContactModel> kContactModelMap = {
      {"stewart_and_trinkle", ContactModel::kStewartAndTrinkle},
      {"anitescu", ContactModel::kAnitescu},
      {"frictionless_spring", ContactModel::kFrictionlessSpring}};
  return kContactModelMap;
}

/**
 * @class LCSFactory
 * @brief Factory class for creating Linear Complementarity Systems (LCS) from
 * multibody plants.
 */
class LCSFactory {
 public:
  LCSFactory(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      LCSFactoryOptions& options);
  /**
   * @brief Constructor for the LCSFactory class.
   *
   * @param plant The standard MultibodyPlant templated on `double`.
   * @param context The context about which to linearize (templated on
   * `double`).
   * @param plant_ad An AutoDiffXd templated MultibodyPlant for gradient
   * calculation.
   * @param context_ad The context about which to linearize (templated on
   * `AutoDiffXd`).
   * @param contact_geoms Vector of geometry pairs defining contact points.
   * @param options Options for LCS creation, including friction properties and
   * contact model.
   */
  LCSFactory(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          contact_geoms,
      const LCSFactoryOptions& options);

  /**
   * @brief Generates a Linear Complementarity System (LCS).
   *
   * @return LCS The resulting Linear Complementarity System.
   */
  LCS GenerateLCS();

  /**
   * @brief Computes the contact Jacobian for a given multibody plant and
   * context.
   *
   * This method calculates the signed distance values and the contact Jacobians
   * for normal and tangential forces at the specified contact points.
   *
   * @return A pair containing the contact Jacobian matrix and a vector of
   * contact points.
   */
  std::pair<MatrixXd, std::vector<VectorXd>> GetContactJacobianAndPoints();

  /**
   * @brief Updates the state and input vectors in the internal context.
   *
   * @param state The state vector.
   * @param input The input vector.
   */
  void UpdateStateAndInput(
      const Eigen::Ref<const drake::VectorX<double>>& state,
      const Eigen::Ref<const drake::VectorX<double>>& input);

  /**
   * @brief Linearizes the dynamics of a multibody plant into a Linear
   * Complementarity System (LCS).
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
   * @param contact_geoms Vector of geometry pairs defining contact points.
   * @param options Options for LCS creation, including friction properties and
   * contact model.
   * @param state The state vector at which to linearize.
   * @param input The input vector at which to linearize.
   * @return LCS The resulting Linear Complementarity System.
   */
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

  /**
   * @brief Creates an LCS by fixing some modes from another LCS.
   *
   * This method modifies the complementarity constraints by ignoring
   * inequalities corresponding to specified active and inactive modes.
   *
   * @param other The original LCS to modify.
   * @param active_lambda_inds Indices for lambda that must be non-zero.
   * @param inactive_lambda_inds Indices for lambda that must be zero.
   * @return LCS The modified Linear Complementarity System.
   */
  static LCS FixSomeModes(const LCS& other, std::set<int> active_lambda_inds,
                          std::set<int> inactive_lambda_inds);

  /**
   * @brief Computes the number of contact variables based on the contact model
   * and number of contacts.
   *
   * @param contact_model The contact model to use.
   * @param num_contacts The number of contact points.
   * @param num_friction_directions The total number of friction directions.
   * @return int The number of contact variables.
   */
  static int GetNumContactVariables(ContactModel contact_model,
                                    int num_contacts,
                                    int num_friction_directions);

  /**
   * @brief Computes the number of contact variables based on the contact model
   * and per-contact friction directions.
   *
   * @param contact_model The contact model to use.
   * @param num_contacts The number of contact points.
   * @param num_friction_directions_per_contact The number of friction
   * directions for each contact.
   * @return int The number of contact variables.
   */
  static int GetNumContactVariables(
      ContactModel contact_model, int num_contacts,
      std::vector<int> num_friction_directions_per_contact);

  /**
   * @brief Computes the number of contact variables based on the LCS options.
   *
   * This is the preferred overload as it encapsulates all contact model and
   * friction configuration in a single options object.
   *
   * @param options The LCS options specifying contact model and friction
   * properties.
   * @return int The number of contact variables.
   */
  static int GetNumContactVariables(const LCSFactoryOptions options);

 private:
  /**
   * @brief Formulates the contact dynamics for the frictionless spring contact
   * model.
   *
   * @param phi Vector of signed distances.
   * @param J_n Contact Jacobian for normal forces.
   * @param qdotNv Matrix relating joint velocities to normal contact
   * velocities.
   * @param spring_stiffness Stiffness of the contact spring.
   * @param[out] M Mass matrix.
   * @param[out] D Damping matrix.
   * @param[out] E Input matrix.
   * @param[out] F Contact force mapping matrix.
   * @param[out] H Complementarity constraint matrix.
   * @param[out] c Constant vector.
   */
  void FormulateFrictionlessSpringContactDynamics(
      const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& qdotNv,
      const double& spring_stiffness, MatrixX<AutoDiffXd>& M, MatrixXd& D,
      MatrixXd& E, MatrixXd& F, MatrixXd& H, VectorXd& c);

  /**
   * @brief Formulates the contact dynamics for the Stewart-Trinkle contact
   * model.
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
   * @param[out] D Damping matrix.
   * @param[out] E Input matrix.
   * @param[out] F Contact force mapping matrix.
   * @param[out] H Complementarity constraint matrix.
   * @param[out] c Constant vector.
   */
  void FormulateStewartTrinkleContactDynamics(
      const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& J_t,
      const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
      const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
      const VectorXd& mu, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
      MatrixXd& F, MatrixXd& H, VectorXd& c);

  /**
   * @brief Formulates the contact dynamics for the Anitescu contact model.
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
   * @param[out] D Damping matrix.
   * @param[out] E Input matrix.
   * @param[out] F Contact force mapping matrix.
   * @param[out] H Complementarity constraint matrix.
   * @param[out] c Constant vector.
   */
  void FormulateAnitescuContactDynamics(
      const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& J_t,
      const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
      const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
      const VectorXd& mu, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
      MatrixXd& F, MatrixXd& H, VectorXd& c);

  /**
   * @brief Computes the contact Jacobian matrices for normal and tangential
   * forces.
   *
   * @param[out] phi Vector of signed distances.
   * @param[out] Jn Contact Jacobian for normal forces.
   * @param[out] Jt Contact Jacobian for tangential forces.
   */
  void ComputeContactJacobian(VectorXd& phi, MatrixXd& Jn, MatrixXd& Jt);

  /**
   * @brief Finds the witness points for each contact pair.
   *
   * @return A pair of vectors containing the witness points on each geometry
   * for each contact pair.
   */
  std::pair<std::vector<VectorXd>, std::vector<VectorXd>> FindWitnessPoints();

  // References to the MultibodyPlant and its contexts
  const drake::multibody::MultibodyPlant<double>& plant_;
  drake::systems::Context<double>& context_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  drake::systems::Context<drake::AutoDiffXd>& context_ad_;
  std::vector<drake::SortedPair<drake::geometry::GeometryId>> contact_pairs_;

  // Configuration options for the LCSFactory
  LCSFactoryOptions options_;

  int n_contacts_;  ///< Number of contact points.
  std::vector<int>
      n_friction_directions_per_contact_;  ///< Number of friction directions.
  std::vector<std::array<double, 3>>
      planar_normal_direction_per_contact_;  ///< Optional normal vector for
                                             ///< planar contact for each
                                             ///< contact.
  ContactModel contact_model_;               ///< The contact model being used.
  int n_q_;       ///< Number of configuration variables.
  int n_v_;       ///< Number of velocity variables.
  int n_x_;       ///< Number of state variables.
  int n_lambda_;  ///< Number of contact force variables.
  int n_u_;       ///< Number of input variables.

  std::vector<double> mu_;  ///< Vector of friction coefficients.
  bool frictionless_;       ///< Flag indicating frictionless contacts.
  double dt_;               ///< Time step.

  VectorXi Jt_row_sizes_;  ///< Row sizes for tangential Jacobian blocks.
};

}  // namespace multibody
}  // namespace c3
