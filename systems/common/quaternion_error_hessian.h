#include <Eigen/Core>
#include <Eigen/Dense>
#include "drake/common/drake_assert.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace c3 {
namespace systems {
namespace common {

/*!
 * Computes the Hessian of the squared angle difference between two quaternions,
 * with respect to the elements of the first.
 * @param quat The first quaternion (size 4 vector).  The Hessian of the squared
 * angle difference is with respect to this quaternion.
 * @param quat_desired The second quaternion (size 4 vector).
 * @return The 4x4 Hessian matrix of the squared angle difference.
*/
Eigen::MatrixXd hessian_of_squared_quaternion_angle_difference(
    const Eigen::VectorXd& quat,
    const Eigen::VectorXd& quat_desired);

} // namespace common
} // namespace systems
} // namespace c3
