#include <chrono>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "systems/common/quaternion_error_hessian.h"
#include "systems/c3_controller.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using drake::systems::DiagramBuilder;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;

namespace c3 {
namespace systems {
namespace test {

class QuaternionCostTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Copy function from c3 controller to test, slightly modified to return Q
  std::vector<MatrixXd> UpdateQuaternionCosts(
    const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_des,
      C3ControllerOptions controller_options) const {
    std::vector<MatrixXd> Q;
    std::vector<MatrixXd> R;
    std::vector<MatrixXd> G;
    std::vector<MatrixXd> U;

    double discount_factor = 1;
    for (int i = 0; i < controller_options.lcs_factory_options.N; i++) {
      Q.push_back(discount_factor *  controller_options.c3_options.Q);
      discount_factor *=  controller_options.c3_options.gamma;
      if (i < controller_options.lcs_factory_options.N) {
        R.push_back(discount_factor *  controller_options.c3_options.R);
        G.push_back(controller_options.c3_options.G);
        U.push_back(controller_options.c3_options.U);
      }
    }  
    Q.push_back(discount_factor * controller_options.c3_options.Q); 

    for (int index : controller_options.quaternion_indices) {
      
      Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4);
      Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
      Eigen::MatrixXd quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd quat_regularizer_2 = quat_des_i * quat_des_i.transpose();

      // Additional regularization term to help with numerical issues
      Eigen::MatrixXd quat_regularizer_3 = 1e-8 * Eigen::MatrixXd::Identity(4, 4);

      double discount_factor = 1;
      for (int i = 0; i < controller_options.lcs_factory_options.N + 1; i++) {
        Q[i].block(index, index, 4, 4) = 
          discount_factor * controller_options.quaternion_weight * 
          (quat_hessian_i + quat_regularizer_1 + 
           controller_options.quaternion_regularizer_fraction * quat_regularizer_2 + quat_regularizer_3);
        discount_factor *= controller_options.c3_options.gamma;
      }
    }
    return Q;
  }
};

TEST_F(QuaternionCostTest, HessianAngleDifferenceTest) {
  // Construct from initializer list.
  VectorXd q1(4);
  q1 << 0.5, 0.5, 0.5, 0.5;

  MatrixXd hessian = common::hessian_of_squared_quaternion_angle_difference(q1, q1);
  std::cout << hessian << std::endl;

  MatrixXd true_hessian(4, 4);
  true_hessian << 6, -2, -2, -2,
                  -2, 6, -2, -2,
                  -2, -2, 6, -2,
                  -2, -2, -2, 6;

  EXPECT_EQ(hessian.rows(), 4);
  EXPECT_EQ(hessian.cols(), 4);
  EXPECT_TRUE(hessian.isApprox(true_hessian, 1e-4));

  VectorXd q2(4);
  q2 << 0.707, 0, 0, 0.707;  

  hessian = common::hessian_of_squared_quaternion_angle_difference(q1, q2);
  std::cout << hessian << std::endl;

  true_hessian << 8.28319,  -2,       -2,        2,
                -2,        2,       -4.28319, -2,
                -2,       -4.28319, 2,       -2,
                2,       -2,       -2,       8.28319;

  EXPECT_EQ(hessian.rows(), 4);
  EXPECT_EQ(hessian.cols(), 4);
  EXPECT_TRUE(hessian.isApprox(true_hessian, 1e-4));
}



TEST_F(QuaternionCostTest, QuaternionCostMatrixTest) {
  C3ControllerOptions controller_options =
    drake::yaml::LoadYamlFile<C3ControllerOptions>(
        "systems/test/quaternion_cost_test.yaml");

  VectorXd q1(4);
  q1 << 0.5, 0.5, 0.5, 0.5;
  VectorXd q2(4);
  q2 << 0.707, 0, 0, 0.707;   

  std::vector<MatrixXd> Q = UpdateQuaternionCosts(q1, q2, controller_options);
    
  for (int i = 0; i < Q.size(); i++) {
    // Check each Q is PSD
    double min_eigenval = Q[i].eigenvalues().real().minCoeff();
    EXPECT_TRUE(min_eigenval >= 0);

    // Check each expected block has been updated
    std::vector<int> start_indices = controller_options.quaternion_indices;
    for (int idx : start_indices) {
      MatrixXd Q_block = Q[i].block(idx, idx, 4, 4);
      MatrixXd Q_diag =  Q_block.diagonal().asDiagonal();
      EXPECT_TRUE(!(Q_block.isApprox(Q_diag)));
    }
  } 

}

} // namespace test
} // namespace systems
} // namespace c3