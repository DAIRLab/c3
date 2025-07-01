#pragma once

#include "drake/common/sorted_pair.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace c3 {
namespace multibody {
/**
 * @brief A class for computing collider properties between two geometries.
 *
 * This class calculates the signed distance and contact frame Jacobian between
 * two geometries in a MultibodyPlant. It provides methods for evaluating
 * these properties under various conditions, including different Jacobian
 * formulations and friction models.
 *
 * @tparam T The scalar type used for computations.
 */
template <typename T>
class GeomGeomCollider {
 public:
  /**
   * @brief Constructor.
   *
   * Constructs a GeomGeomCollider object, associating it with a specific
   * MultibodyPlant and a pair of geometries.
   *
   * @param plant The MultibodyPlant containing the geometries of interest.
   * @param geometry_pair A sorted pair of GeometryIds, uniquely identifying
   *        the two geometries for collision evaluation. The order within the
   *        pair is determined by drake::SortedPair.
   */
  GeomGeomCollider(
      const drake::multibody::MultibodyPlant<T>& plant,
      const drake::SortedPair<drake::geometry::GeometryId> geometry_pair);

  /**
   * @brief Calculates the signed distance and contact frame Jacobian using a
   *        polytope representation of friction.
   *
   * This method approximates the friction cone as a polytope. The Jacobian is
   * ordered as [J_n; J_t1; ...; J_tN], where J_n is the Jacobian of the
   * contact normal direction, and J_t1 through J_tN are the Jacobians of the
   * tangent directions that define the polytope. The shape of the Jacobian
   * matrix is (2*num_friction_directions + 1) x (nq or nv), depending on the
   * choice of JacobianWrtVariable.
   *
   * @param context The context for the MultibodyPlant.
   * @param num_friction_directions The number of friction directions to use
   *        in the polytope approximation. Must be greater than 1.
   * @param wrt An enum specifying whether the Jacobian should be computed with
   *        respect to configuration variables (q) or velocity variables (v).
   *        Defaults to JacobianWrtVariable::kV.
   * @return A pair containing the signed distance as a scalar and the
   *         contact frame Jacobian as a matrix.
   */
  std::pair<T, drake::MatrixX<T>> EvalPolytope(
      const drake::systems::Context<T>& context, int num_friction_directions,
      drake::multibody::JacobianWrtVariable wrt =
          drake::multibody::JacobianWrtVariable::kV);

  /**
   * @brief Calculates the signed distance and contact frame Jacobian for a 2D
   *        planar problem.
   *
   * In a 2D planar problem, the contact frame consists of a normal direction
   * and a single tangent direction. This method computes the Jacobian in the
   * order [J_n; +J_t; -J_t], where J_n is the Jacobian of the contact normal
   * direction, and J_t is the Jacobian of the tangent direction. The shape of
   * the Jacobian matrix is 3 x (nq or nv), depending on the choice of
   * JacobianWrtVariable.
   *
   * @param context The context for the MultibodyPlant.
   * @param planar_normal The normal vector to the planar system, defining the
   *        plane in which the system operates.
   * @param wrt An enum specifying whether the Jacobian should be computed with
   *        respect to configuration variables (q) or velocity variables (v).
   *        Defaults to JacobianWrtVariable::kV.
   * @return A pair containing the signed distance as a scalar and the
   *         contact frame Jacobian as a matrix.
   */
  std::pair<T, drake::MatrixX<T>> EvalPlanar(
      const drake::systems::Context<T>& context,
      const Eigen::Vector3d& planar_normal,
      drake::multibody::JacobianWrtVariable wrt =
          drake::multibody::JacobianWrtVariable::kV);

  /**
   * @brief Returns the positions, expressed in the World frame, of the
   *        closest points on each Geometry.
   *
   * The distance between the two points should match the distance returned by
   * Eval. The order of the points returned depends on the order in which the
   * geometries were added to the MultibodyPlant.
   *
   * @param context The context for the MultibodyPlant.
   * @return A pair of positions (as 3D vectors) in sorted order, as
   *         determined by SortedPair. The first element of the pair is the
   *         closest point on geometry A, and the second element is the
   *         closest point on geometry B.
   */
  std::pair<drake::VectorX<double>, drake::VectorX<double>> CalcWitnessPoints(
      const drake::systems::Context<T>& context);

  /**
   * @brief Computes a basis for contact forces in the world frame.
   *
   * Depending on the number of friction directions, this method constructs
   * either a planar (2D) or polytope (3D) basis for the contact forces at the
   * collision point, expressed in the world frame. For planar contact
   * (num_friction_directions < 1), the basis is constructed from the contact
   * normal and the provided planar normal. For 3D contact, a polytope basis is
   * generated and rotated to align with the contact normal.
   *
   * @param context The context for the MultibodyPlant.
   * @param num_friction_directions The number of friction directions for the
   * polytope approximation. If less than 1, a planar basis is used.
   * @param planar_normal The normal vector defining the plane for planar
   * contact (default: {0, 1, 0}).
   * @return A matrix whose rows form an orthonormal basis for the contact
   * forces in the world frame.
   */
  Eigen::Matrix<double, Eigen::Dynamic, 3> CalcForceBasisInWorldFrame(
      const drake::systems::Context<T>& context, int num_friction_directions,
      const Eigen::Vector3d& planar_normal = {0, 1, 0}) const;

 private:
  /**
   * @brief A struct to hold the results of a geometry query.
   *
   * This struct contains the signed distance pair, the frame IDs of the two
   * geometries, the frames themselves, and the positions of the closest
   * points on each geometry, expressed in their respective frames.
   */
  struct GeometryQueryResult {
    /**
     * @brief The signed distance pair between the two geometries.
     */
    drake::geometry::SignedDistancePair<T> signed_distance_pair;
    /**
     * @brief The FrameId of the first frame.
     */
    const drake::geometry::FrameId frame_A_id;
    /**
     * @brief The FrameId of the second frame.
     */
    const drake::geometry::FrameId frame_B_id;
    /**
     * @brief A reference to the first frame.
     */
    const drake::multibody::Frame<T>& frameA;
    /**
     * @brief A reference to the second frame.
     */
    const drake::multibody::Frame<T>& frameB;
    /**
     * @brief The position of the closest point on geometry A, expressed in
     * frame A.
     */
    Eigen::Vector3d p_ACa;
    /**
     * @brief The position of the closest point on geometry B, expressed in
     * frame B.
     */
    Eigen::Vector3d p_BCb;
  };

  /**
   * @brief Internal helper function for EvalPolytope and EvalPlanar.
   *
   * This function performs the core computation of the signed distance and
   * contact frame Jacobian, given a specified force basis and rotation matrix.
   *
   * @param context The context for the MultibodyPlant.
   * @param query_result The struct containing the results of the geometry
   *        query.
   * @param force_basis A matrix whose columns form an orthonormal basis for
   *        the contact forces. For a 3D contact, this will typically be a 3x3
   *        identity matrix. For a 2D planar contact, this will be a 3x2
   *        matrix whose columns are the contact normal and tangent vectors.
   * @param wrt An enum specifying whether the Jacobian should be computed with
   *        respect to configuration variables (q) or velocity variables (v).
   * @param R_WC A rotation matrix from the world frame to the contact frame.
   * @return A pair containing the signed distance as a scalar and the
   *         contact frame Jacobian as a matrix.
   */
  std::pair<T, drake::MatrixX<T>> DoEval(
      const drake::systems::Context<T>& context,
      const GeometryQueryResult query_result,
      Eigen::Matrix<double, Eigen::Dynamic, 3> force_basis,
      drake::multibody::JacobianWrtVariable wrt,
      const drake::math::RotationMatrix<T>& R_WC);

  /**
   * @brief Computes the force basis for a polytope approximation of a friction
   * cone.
   *
   * This function calculates a set of vectors that define the
   * directions along which contact forces can be applied. These vectors are
   * used to approximate a friction cone as a polytope. The number of vectors
   * determines the fidelity of the approximation.
   *
   * @param num_friction_directions The number of friction directions to use in
   *        the polytope approximation. This value determines the number of
   *        edges in the polytope and must be greater than 1.
   *
   * @return A matrix whose columns form  basis vectors for the
   *         contact forces. The first column is the contact normal, and the
   *         remaining columns are tangent vectors that define the edges of the
   *         polytope.
   */
  Eigen::Matrix<double, Eigen::Dynamic, 3> ComputePolytopeForceBasis(
      const int num_friction_directions) const;

  /**
   * @brief Computes the force basis for a 2D planar problem.
   *
   * Given a contact normal and a planar normal, this function computes an
   * orthonormal basis for the contact forces in the 2D plane.
   *
   * @param contact_normal The normal vector to the contact surface.
   * @param planar_normal The normal vector to the planar system, defining the
   *        plane in which the system operates.
   * @return A 3x3 matrix whose columns form an orthonormal basis for the
   *         contact forces.
   */
  Eigen::Matrix3d ComputePlanarForceBasis(
      const Eigen::Vector3d& contact_normal,
      const Eigen::Vector3d& planar_normal) const;

  /**
   * @brief Gets the geometry query result.
   *
   * This function queries the MultibodyPlant for the signed distance and
   * closest points between the two geometries.
   *
   * @param context The context for the MultibodyPlant.
   * @return A GeometryQueryResult struct containing the results of the query.
   */
  GeometryQueryResult GetGeometryQueryResult(
      const drake::systems::Context<T>& context) const;

  /**
   * @brief A reference to the MultibodyPlant containing the geometries.
   */
  const drake::multibody::MultibodyPlant<T>& plant_;
  /**
   * @brief The GeometryId of the first geometry in the collision pair.
   */
  const drake::geometry::GeometryId geometry_id_A_;
  /**
   * @brief The GeometryId of the second geometry in the collision pair.
   */
  const drake::geometry::GeometryId geometry_id_B_;
};

}  // namespace multibody
}  // namespace c3

// Explicitly instantiate the template class for double precision.  This is
// needed because the class is defined in a header file, and we want to make
// sure that the implementation is available in a single translation unit.
extern template class c3::multibody::GeomGeomCollider<double>;
