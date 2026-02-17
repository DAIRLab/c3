#include <chrono>
#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <drake/multibody/parsing/parser.h>

#include "systems/c3_controller.h"
#include "systems/common/quaternion_error_hessian.h"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "core/c3.h"

using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::systems::DiagramBuilder;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using drake::systems::DiagramBuilder;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;

namespace c3 {
namespace systems {
namespace test {

class QuaternionCostTest : public ::testing::Test {
 protected:

  C3ControllerOptions controller_options_;
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  C3Controller* c3_controller_;

  void SetUp() override {

    DiagramBuilder<double> builder;
    auto [plant, scene_graph] =
        AddMultibodyPlantSceneGraph(&builder, 0);
    Parser parser(&plant, &scene_graph);    
    parser.AddModels("systems/test/resources/cube.sdf");
    parser.AddModels("systems/test/resources/plate.sdf");
    plant.Finalize();

    controller_options_ =
      drake::yaml::LoadYamlFile<C3ControllerOptions>(
          "systems/test/resources/quaternion_cost_test.yaml");

    vector<MatrixXd> Q_dummy;
    vector<MatrixXd> R_dummy;
    vector<MatrixXd> G_dummy;
    vector<MatrixXd> U_dummy;

    double discount_factor = 1;
    for (int i = 0; i < controller_options_.lcs_factory_options.N; i++) {
      Q_dummy.push_back(discount_factor * controller_options_.c3_options.Q);
      discount_factor *= controller_options_.c3_options.gamma;
      if (i < controller_options_.lcs_factory_options.N) {
        R_dummy.push_back(discount_factor * controller_options_.c3_options.R);
        G_dummy.push_back(controller_options_.c3_options.G);
        U_dummy.push_back(controller_options_.c3_options.U);
      }
    }
    Q_dummy.push_back(discount_factor * controller_options_.c3_options.Q);

    C3::CostMatrices dummy_costs = C3::CostMatrices(Q_dummy, R_dummy, G_dummy, U_dummy);
    
    c3_controller_ = builder.AddSystem<C3Controller>(plant, dummy_costs, controller_options_);
    diagram_ = builder.Build();
  }


};

TEST_F(QuaternionCostTest, HessianAngleDifferenceTest) {
  // Construct from initializer list.
  VectorXd q1(4);
  q1 << 0.5, 0.5, 0.5, 0.5;

  MatrixXd hessian = common::hessian_of_squared_quaternion_angle_difference(q1, q1);

  MatrixXd true_hessian(4, 4);
  true_hessian << 6, -2, -2, -2, -2, 6, -2, -2, -2, -2, 6, -2, -2, -2, -2, 6;

  EXPECT_EQ(hessian.rows(), 4);
  EXPECT_EQ(hessian.cols(), 4);
  EXPECT_TRUE(hessian.isApprox(true_hessian, 1e-4));

  VectorXd q2(4);
  q2 << 0.707, 0, 0, 0.707;

  hessian = common::hessian_of_squared_quaternion_angle_difference(q1, q2);

  true_hessian << 8.28319, -2, -2, 2, -2, 2, -4.28319, -2, -2, -4.28319, 2, -2,
      2, -2, -2, 8.28319;

  EXPECT_EQ(hessian.rows(), 4);
  EXPECT_EQ(hessian.cols(), 4);
  EXPECT_TRUE(hessian.isApprox(true_hessian, 1e-4));
}

TEST_F(QuaternionCostTest, QuaternionCostMatrixTest) {

  VectorXd q1(4);
  q1 << 0.5, 0.5, 0.5, 0.5;
  VectorXd q2(4);
  q2 << 0.707, 0, 0, 0.707;

  c3_controller_->UpdateQuaternionCosts(q1, q2);

  C3::CostMatrices output = c3_controller_->GetCostMatrices();
  vector<MatrixXd> Q = output.Q;
    
  for (int i = 0; i < Q.size(); i++) {
    // Check each Q is PSD
    double min_eigenval = Q[i].eigenvalues().real().minCoeff();
    EXPECT_TRUE(min_eigenval >= 0);

    // Check each expected block has been updated
    std::vector<int> start_indices;
    start_indices.push_back(5); // Index of quaternion
    
    for (int idx : start_indices) {
      MatrixXd Q_block = Q[i].block(idx, idx, 4, 4);
      MatrixXd Q_diag = Q_block.diagonal().asDiagonal();
      EXPECT_TRUE(!(Q_block.isApprox(Q_diag)));
    }
  }
}

TEST_F(QuaternionCostTest, HessianSmallAngleDifferenceTest) {
  VectorXd epsilon1(4);
  epsilon1 << 0.001, 0, 0, 0;

  VectorXd epsilon2(4);
  epsilon2 << 0, 0.001, 0, 0;

  VectorXd epsilon3(4);
  epsilon3 << 0, 0, 0.001, 0;

  VectorXd epsilon4(4);
  epsilon4 << 0, 0, 0, 0.001;

  for (int i = 0; i < 10; i++) {
    double roll = 0.6 * i;
    double pitch = -0.13 * i;
    double yaw = 0.44 * i;

    Eigen::Quaterniond q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                           Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Vector4d q_vec(q.w(), q.x(), q.y(), q.z());

    MatrixXd hessian =
        common::hessian_of_squared_quaternion_angle_difference(q_vec, q_vec);

    MatrixXd peturbed_hessian =
        common::hessian_of_squared_quaternion_angle_difference(
            q_vec, q_vec + epsilon1);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon2);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon3);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon4);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon1 + epsilon2);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon2 + epsilon3);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon3 + epsilon4);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon4 + epsilon1);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon3 + epsilon1);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);

    peturbed_hessian = common::hessian_of_squared_quaternion_angle_difference(
        q_vec, q_vec + epsilon4 + epsilon2);
    EXPECT_TRUE(hessian.isApprox(peturbed_hessian, 1e-2))
        << "\n hessian:\n"
        << hessian << "\n perturbed_hessian:\n"
        << peturbed_hessian << "\n difference:\n"
        << (hessian - peturbed_hessian);
  }
}
}  // namespace test
}  // namespace systems
}  // namespace c3
