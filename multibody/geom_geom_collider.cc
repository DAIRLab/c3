#include "multibody/geom_geom_collider.h"

#include <iostream>

#include "drake/math/rotation_matrix.h"

using drake::EigenPtr;
using drake::MatrixX;
using drake::VectorX;
using drake::geometry::GeometryId;
using drake::geometry::SignedDistancePair;
using drake::multibody::JacobianWrtVariable;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using Eigen::Matrix;
using Eigen::Vector3d;

namespace c3 {
namespace multibody {

template <typename T>
GeomGeomCollider<T>::GeomGeomCollider(
    const drake::multibody::MultibodyPlant<T>& plant,
    const drake::SortedPair<drake::geometry::GeometryId> geometry_pair)
    : plant_(plant),
      geometry_id_A_(geometry_pair.first()),
      geometry_id_B_(geometry_pair.second()) {}

template <typename T>
GeomGeomCollider<T>::GeometryQueryResult
GeomGeomCollider<T>::GetGeometryQueryResult(const Context<T>& context) const {
  // Access the geometry query object from the plant's geometry query port.
  const auto& query_port = plant_.get_geometry_query_input_port();
  const auto& query_object =
      query_port.template Eval<drake::geometry::QueryObject<T>>(context);

  // Compute the signed distance pair between the two geometries.
  const SignedDistancePair<T> signed_distance_pair =
      query_object.ComputeSignedDistancePairClosestPoints(geometry_id_A_,
                                                          geometry_id_B_);

  // Access the geometry inspector from the query object.
  const auto& inspector = query_object.inspector();
  const auto frame_A_id = inspector.GetFrameId(geometry_id_A_);
  const auto frame_B_id = inspector.GetFrameId(geometry_id_B_);

  // Get the frames associated with the geometry ids
  const auto& frameA = plant_.GetBodyFromFrameId(frame_A_id)->body_frame();
  const auto& frameB = plant_.GetBodyFromFrameId(frame_B_id)->body_frame();

  // Get the poses of the contact points in their respective frames.
  const Vector3d& p_ACa =
      inspector.GetPoseInFrame(geometry_id_A_).template cast<T>() *
      signed_distance_pair.p_ACa;
  const Vector3d& p_BCb =
      inspector.GetPoseInFrame(geometry_id_B_).template cast<T>() *
      signed_distance_pair.p_BCb;

  return GeometryQueryResult{signed_distance_pair,
                             frame_A_id,
                             frame_B_id,
                             frameA,
                             frameB,
                             p_ACa,
                             p_BCb};
}

template <typename T>
std::pair<T, MatrixX<T>> GeomGeomCollider<T>::DoEval(
    const Context<T>& context,
    const GeomGeomCollider<T>::GeometryQueryResult query_result,
    Matrix<double, Eigen::Dynamic, 3> force_basis, JacobianWrtVariable wrt,
    const drake::math::RotationMatrix<T>& R_WC) {
  // Determine Jacobian dimensions
  const int n_cols = (wrt == JacobianWrtVariable::kV) ? plant_.num_velocities()
                                                      : plant_.num_positions();
  Matrix<double, 3, Eigen::Dynamic> Jv_WCa(3, n_cols);
  Matrix<double, 3, Eigen::Dynamic> Jv_WCb(3, n_cols);

  // Compute Jacobians for both contact points
  plant_.CalcJacobianTranslationalVelocity(
      context, wrt, query_result.frameA, query_result.p_ACa,
      plant_.world_frame(), plant_.world_frame(), &Jv_WCa);
  plant_.CalcJacobianTranslationalVelocity(
      context, wrt, query_result.frameB, query_result.p_BCb,
      plant_.world_frame(), plant_.world_frame(), &Jv_WCb);

  // Compute final Jacobian: J = force_basis * R_WC^T * (Jv_WCa - Jv_WCb)
  auto J = force_basis * R_WC.matrix().transpose() * (Jv_WCa - Jv_WCb);

  return std::pair<T, MatrixX<T>>(query_result.signed_distance_pair.distance,
                                  J);
}

template <typename T>
std::pair<T, MatrixX<T>> GeomGeomCollider<T>::EvalPolytope(
    const Context<T>& context, int num_friction_directions,
    JacobianWrtVariable wrt) {
  if (num_friction_directions < 1) {
    throw std::runtime_error(fmt::format(
        "GeomGeomCollider cannot specify %d friction direction unless "
        "using EvalPlanar.",
        num_friction_directions));
  }

  auto polytope_force_bases =
      ComputePolytopeForceBasis(num_friction_directions);

  // Get geometry query result to access contact normal
  const auto query_result = GetGeometryQueryResult(context);

  // Create rotation matrix from contact normal
  auto R_WC = drake::math::RotationMatrix<T>::MakeFromOneVector(
      query_result.signed_distance_pair.nhat_BA_W, 0);

  return DoEval(context, query_result, polytope_force_bases, wrt, R_WC);
}

template <typename T>
Matrix<double, Eigen::Dynamic, 3>
GeomGeomCollider<T>::ComputePolytopeForceBasis(
    const int num_friction_directions) const {
  // Build friction basis
  Matrix<double, Eigen::Dynamic, 3> force_basis(2 * num_friction_directions + 1,
                                                3);
  force_basis.row(0) << 1, 0, 0;

  for (int i = 0; i < num_friction_directions; i++) {
    double theta = (M_PI * i) / num_friction_directions;
    force_basis.row(2 * i + 1) = Vector3d(0, cos(theta), sin(theta));
    force_basis.row(2 * i + 2) = -force_basis.row(2 * i + 1);
  }
  return force_basis;
}

template <typename T>
std::pair<T, MatrixX<T>> GeomGeomCollider<T>::EvalPlanar(
    const Context<T>& context, const Vector3d& planar_normal,
    JacobianWrtVariable wrt) {
  // Get geometry query result to access contact normal
  const auto query_result = GetGeometryQueryResult(context);

  // Compute the planar force basis using the contact normal and planar normal
  auto planar_force_basis = ComputePlanarForceBasis(
      query_result.signed_distance_pair.nhat_BA_W, planar_normal);

  // For planar case, use identity rotation since force basis is already in
  // world frame
  auto R_WC = drake::math::RotationMatrix<T>::Identity();

  return DoEval(context, query_result, planar_force_basis, wrt, R_WC);
}

template <typename T>
Eigen::Matrix3d GeomGeomCollider<T>::ComputePlanarForceBasis(
    const Eigen::Vector3d& contact_normal,
    const Eigen::Vector3d& planar_normal) const {
  Eigen::Matrix3d force_basis = Eigen::Matrix3d::Zero();

  // First row is the contact normal, projected to the plane
  force_basis.row(0) =
      contact_normal - planar_normal * planar_normal.dot(contact_normal);
  force_basis.row(0).normalize();

  // Second row is the cross product between contact normal and planar normal
  force_basis.row(1) = contact_normal.cross(planar_normal);
  force_basis.row(1).normalize();
  force_basis.row(2) = -force_basis.row(1);

  return force_basis;
}

template <typename T>
std::pair<VectorX<double>, VectorX<double>>
GeomGeomCollider<T>::CalcWitnessPoints(const Context<double>& context) {
  // Get common geometry query results
  const auto query_result = GetGeometryQueryResult(context);

  // Calculate world positions of contact points
  Vector3d p_WCa = Vector3d::Zero();
  Vector3d p_WCb = Vector3d::Zero();
  plant_.CalcPointsPositions(context, query_result.frameA, query_result.p_ACa,
                             plant_.world_frame(), &p_WCa);
  plant_.CalcPointsPositions(context, query_result.frameB, query_result.p_BCb,
                             plant_.world_frame(), &p_WCb);
  return std::pair<VectorX<double>, VectorX<double>>(p_WCa, p_WCb);
}

}  // namespace multibody
}  // namespace c3

template class c3::multibody::GeomGeomCollider<double>;