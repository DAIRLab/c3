#pragma once

#include <map>
#include <set>

#include "core/c3_options.h"
#include "core/lcs.h"

#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

using drake::AutoDiffXd;
using drake::MatrixX;
using Eigen::MatrixXd;
using Eigen::VectorXd;

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
struct ContactModelMap : public std::map<std::string, ContactModel> {
  ContactModelMap() {
    this->operator[]("stewart_and_trinkle") = ContactModel::kStewartAndTrinkle;
    this->operator[]("anitescu") = ContactModel::kAnitescu;
    this->operator[]("frictionless_spring") = ContactModel::kFrictionlessSpring;
  };
  ~ContactModelMap() {}
};

/**
 * @class LCSFactory
 * @brief Factory class for creating Linear Complementarity Systems (LCS) from
 * multibody plants.
 */
class LCSFactory {
 public:
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
   * @return LCS The resulting Linear Complementarity System.
   */
  static LCS LinearizePlantToLCS(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      const drake::systems::Context<drake::AutoDiffXd>& context_ad,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          contact_geoms,
      const LCSOptions& options);

  /**
   * @brief Computes the contact Jacobian for a given multibody plant and
   * context.
   *
   * This method calculates the signed distance values and the contact Jacobians
   * for normal and tangential forces at the specified contact points.
   *
   * @param plant The standard MultibodyPlant templated on `double`.
   * @param context The context for the plant (templated on `double`).
   * @param contact_geoms Vector of geometry pairs defining contact points.
   * @param options Options for LCS creation, including friction properties and
   * contact model.
   * @return A pair containing the contact Jacobian matrix and a vector of
   * contact points.
   */
  static std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>
  ComputeContactJacobian(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::systems::Context<double>& context,
      const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
          contact_geoms,
      const LCSOptions& options);

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
   * and options.
   *
   * @param contact_model The contact model to use.
   * @param num_contacts The number of contact points.
   * @param num_friction_directions The number of friction directions.
   * @param frictionless Whether the contacts are frictionless.
   * @return int The number of contact variables.
   */
  static int GetNumContactVariables(ContactModel contact_model,
                                    int num_contacts,
                                    int num_friction_directions);

  /**
   * @brief Computes the number of contact variables based on the LCS options.
   *
   * @param options The LCS options specifying contact model and friction
   * properties.
   * @return int The number of contact variables.
   */
  static int GetNumContactVariables(const LCSOptions options);

 private:
  static void FormulateFrictionlessSpringContactDynamics(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::systems::Context<double>& context, const int& n_q,
      const int& n_v, const int& n_contacts, const double& dt,
      const VectorXd& phi, const MatrixXd& J_n, const MatrixXd& qdotNv,
      const double& spring_stiffness, MatrixX<AutoDiffXd>& M, MatrixXd& D,
      MatrixXd& E, MatrixXd& F, MatrixXd& H, VectorXd& c);

  /**
   * @brief Formulates the contact dynamics for the Stewart-Trinkle contact
   * model.
   *
   * Computes the matrices and vectors necessary for the Anitescu contact
   * model, including mappings between state and forces, complementarity
   * constraints, and other dynamics-related quantities. This function is
   * intended for internal use to improve code organization and readability.
   */
  static void FormulateStewartTrinkleContactDynamics(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::systems::Context<double>& context, const int& n_q,
      const int& n_v, const int& n_u, const int& n_contacts,
      const int& num_friction_directions, const double& dt, const VectorXd& phi,
      const MatrixXd& J_n, const MatrixXd& J_t, const MatrixXd& Jf_q,
      const MatrixXd& Jf_v, const MatrixXd& Jf_u, const VectorXd& d_v,
      const MatrixXd& vNqdot, const MatrixXd& qdotNv, const VectorXd& mu,
      MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E, MatrixXd& F,
      MatrixXd& H, VectorXd& c);

  /**
   * @brief Formulates the contact dynamics for the Anitescu contact model.
   *
   * Computes the matrices and vectors necessary for the Anitescu contact model,
   * including mappings between state and forces, complementarity constraints,
   * and other dynamics-related quantities. This function is intended for
   * internal use to improve code organization and readability.
   */
  static void FormulateAnitescuContactDynamics(
      const drake::multibody::MultibodyPlant<double>& plant,
      const drake::systems::Context<double>& context, const int& n_q,
      const int& n_v, const int& n_contacts, const int& n_lambda,
      const int& num_friction_directions, const double& dt, const VectorXd& phi,
      const MatrixXd& J_n, const MatrixXd& J_t, const MatrixXd& Jf_q,
      const MatrixXd& Jf_v, const MatrixXd& Jf_u, const VectorXd& d_v,
      const MatrixXd& vNqdot, const MatrixXd& qdotNv, const VectorXd& mu,
      MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E, MatrixXd& F,
      MatrixXd& H, VectorXd& c);
};

}  // namespace multibody
}  // namespace c3
