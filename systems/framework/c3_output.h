#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::VectorXf;

namespace c3 {
namespace systems {

/// Used for outputting c3 solutions and intermediate variables for debugging
/// purposes

class C3Output {
 public:
  struct C3Solution {
    C3Solution() = default;
    C3Solution(int n_x, int n_lambda, int n_u, int N) {
      x_sol_ = MatrixXf::Zero(n_x, N);
      lambda_sol_ = MatrixXf::Zero(n_lambda, N);
      u_sol_ = MatrixXf::Zero(n_u, N);
      time_vector_ = VectorXf::Zero(N);
    };

    // Shape is (variable_vector_size, knot_points)
    Eigen::VectorXf time_vector_;
    Eigen::MatrixXf x_sol_;
    Eigen::MatrixXf lambda_sol_;
    Eigen::MatrixXf u_sol_;
  };

  struct C3Intermediates {
    C3Intermediates() = default;
    C3Intermediates(int n_x, int n_lambda, int n_u, int N) {
      z_ = MatrixXf::Zero(n_x + n_lambda + n_u, N);
      delta_ = MatrixXf::Zero(n_x + n_lambda + n_u, N);
      w_ = MatrixXf::Zero(n_x + n_lambda + n_u, N);
      time_vector_ = VectorXf::Zero(N);
    };
    C3Intermediates(int n_z, int N) {
      z_ = MatrixXf::Zero(n_z, N);
      delta_ = MatrixXf::Zero(n_z, N);
      w_ = MatrixXf::Zero(n_z, N);
      time_vector_ = VectorXf::Zero(N);
    };

    // Shape is (variable_vector_size, knot_points)
    Eigen::VectorXf time_vector_;
    Eigen::MatrixXf z_;
    Eigen::MatrixXf delta_;
    Eigen::MatrixXf w_;
  };

  C3Output() = default;
  C3Output(C3Solution c3_sol, C3Intermediates c3_intermediates);

  // explicit C3Output(const lcmt_c3_output& traj);

  virtual ~C3Output() = default;

 private:
  C3Solution c3_solution_;
  C3Intermediates c3_intermediates_;
};
}  // namespace systems
}  // namespace c3
