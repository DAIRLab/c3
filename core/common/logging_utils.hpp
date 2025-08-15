#include <sstream>
#include <string>

#include <Eigen/Dense>

/**
 * Converts an Eigen matrix to a string representation.
 * This function is useful for logging or debugging purposes.
 *
 * @tparam Derived The type of the Eigen matrix.
 * @param mat The Eigen matrix to convert.
 * @return A string representation of the matrix.
 */
template <typename Derived>
std::string EigenToString(const Eigen::MatrixBase<Derived>& mat) {
  std::stringstream ss;
  ss << mat;
  return ss.str();
}