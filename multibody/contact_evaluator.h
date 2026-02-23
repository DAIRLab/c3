#pragma once

#include <memory>

#include <Eigen/Dense>

#include "geom_geom_collider.h"

#include "drake/common/sorted_pair.h"
#include "drake/geometry/geometry_ids.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace c3 {
namespace multibody {

/**
 * @brief Abstract base class for evaluating contact constraints.
 *
 * ContactEvaluator provides an interface for computing signed distance and
 * contact Jacobians for different friction models (planar vs polytope).
 */
template <typename T>
class ContactEvaluator {
 public:
  ContactEvaluator(
      const drake::multibody::MultibodyPlant<T>& plant,
      const drake::SortedPair<drake::geometry::GeometryId>& contact_pair)
      : collider_(plant, contact_pair) {}

  virtual ~ContactEvaluator() = default;

  /**
   * @brief Evaluates the signed distance and contact Jacobian.
   *
   * @param context The context for the MultibodyPlant.
   * @param wrt Whether to compute Jacobian w.r.t. q or v.
   * @return Pair of (signed_distance, contact_jacobian)
   */
  virtual std::pair<T, drake::MatrixX<T>> Eval(
      const drake::systems::Context<T>& context,
      drake::multibody::JacobianWrtVariable wrt =
          drake::multibody::JacobianWrtVariable::kV) const = 0;

  /**
   * @brief Returns the number of friction directions for this contact model.
   */
  virtual int GetNumFrictionDirections() const = 0;

  /**
   * @brief Computes the force basis for the contact constraint.
   *
   * For polytope contacts, returns the friction cone approximation basis.
   * For planar contacts, returns the planar friction basis.
   *
   * @param context The context for the MultibodyPlant.
   * @return Matrix where each row is a force direction in world frame.
   */
  virtual Eigen::Matrix<double, Eigen::Dynamic, 3> CalcForceBasis(
      const drake::systems::Context<T>& context) const = 0;

  /**
   * @brief Computes witness points (closest points on each geometry).
   *
   * @param context The context for the MultibodyPlant.
   * @return Pair of witness points in world frame.
   */
  std::pair<Eigen::Vector3d, Eigen::Vector3d> CalcWitnessPoints(
      const drake::systems::Context<T>& context) const {
    return collider_.CalcWitnessPoints(context);
  }

 protected:
  mutable GeomGeomCollider<T> collider_;
};

/**
 * @brief Contact evaluator for polytope friction cone approximation.
 */
template <typename T>
class PolytopeContactEvaluator : public ContactEvaluator<T> {
 public:
  PolytopeContactEvaluator(
      const drake::multibody::MultibodyPlant<T>& plant,
      const drake::SortedPair<drake::geometry::GeometryId>& contact_pair,
      int num_friction_directions)
      : ContactEvaluator<T>(plant, contact_pair),
        num_friction_directions_(num_friction_directions) {
    DRAKE_DEMAND(num_friction_directions_ > 1);
  }

  std::pair<T, drake::MatrixX<T>> Eval(
      const drake::systems::Context<T>& context,
      drake::multibody::JacobianWrtVariable wrt =
          drake::multibody::JacobianWrtVariable::kV) const override {
    return this->collider_.EvalPolytope(context, num_friction_directions_, wrt);
  }

  int GetNumFrictionDirections() const override {
    return num_friction_directions_;
  }

  Eigen::Matrix<double, Eigen::Dynamic, 3> CalcForceBasis(
      const drake::systems::Context<T>& context) const override {
    return this->collider_.CalcForceBasisInWorldFrame(context,
                                                      num_friction_directions_);
  }

 private:
  int num_friction_directions_;
};

/**
 * @brief Contact evaluator for planar (2D) friction.
 */
template <typename T>
class PlanarContactEvaluator : public ContactEvaluator<T> {
 public:
  PlanarContactEvaluator(
      const drake::multibody::MultibodyPlant<T>& plant,
      const drake::SortedPair<drake::geometry::GeometryId>& contact_pair,
      const Eigen::Vector3d& planar_normal)
      : ContactEvaluator<T>(plant, contact_pair),
        planar_normal_(planar_normal) {
    // Validate unit vector
    DRAKE_DEMAND(std::abs(planar_normal_.norm() - 1.0) < 1e-6);
  }

  std::pair<T, drake::MatrixX<T>> Eval(
      const drake::systems::Context<T>& context,
      drake::multibody::JacobianWrtVariable wrt =
          drake::multibody::JacobianWrtVariable::kV) const override {
    return this->collider_.EvalPlanar(context, planar_normal_, wrt);
  }

  int GetNumFrictionDirections() const override {
    return 1;  // Planar contact has 1 friction direction
  }

  Eigen::Matrix<double, Eigen::Dynamic, 3> CalcForceBasis(
      const drake::systems::Context<T>& context) const override {
    // For planar contact, pass num_friction_directions = 0 to trigger planar
    // mode
    return this->collider_.CalcForceBasisInWorldFrame(context, 0,
                                                      planar_normal_);
  }

 private:
  Eigen::Vector3d planar_normal_;
};

}  // namespace multibody
}  // namespace c3

// Explicitly instantiate for double
extern template class c3::multibody::ContactEvaluator<double>;
extern template class c3::multibody::PolytopeContactEvaluator<double>;
extern template class c3::multibody::PlanarContactEvaluator<double>;