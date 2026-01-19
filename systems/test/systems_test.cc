// systems_test.cc
// Unit tests for LCS-based systems and controllers in the c3 project.

#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>
#include <gtest/gtest.h>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "core/test/c3_cartpole_problem.hpp"
#include "multibody/lcs_factory.h"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/framework/system_output.h"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"

using c3::multibody::LCSContactDescription;
using c3::multibody::LCSFactory;
using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;
using c3::systems::C3Output;
using c3::systems::C3StatePredictionJoint;
using c3::systems::LCSSimulator;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::DiscreteValues;
using drake::systems::SystemOutput;
using Eigen::VectorX;

namespace c3 {
namespace systems {
namespace test {

// Test fixture for LCSSimulator
class LCSSimulatorTest : public ::testing::Test, public C3CartpoleProblem {
 protected:
  LCSSimulatorTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {}

  void SetUp() override {
    context_ = simulator_.CreateDefaultContext();
    output_ = simulator_.AllocateOutput();

    // Set up dummy inputs for LCS, state, and action
    simulator_.get_input_port(0).FixValue(context_.get(), *pSystem);
    simulator_.get_input_port(1).FixValue(context_.get(), x0);
    VectorX<double> ones = VectorX<double>::Ones(pSystem->num_inputs());
    simulator_.get_input_port(2).FixValue(context_.get(), ones);
  }

  LCSSimulator simulator_{*pSystem};
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
};

TEST_F(LCSSimulatorTest, Initialization) {
  // Check if the simulator is initialized correctly.
  EXPECT_EQ(simulator_.get_input_port_state().size(), pSystem->num_states());
  EXPECT_EQ(simulator_.get_input_port_action().size(), pSystem->num_inputs());
  EXPECT_EQ(simulator_.get_output_port_next_state().size(),
            pSystem->num_states());

  // Test alternate constructor
  LCSSimulator simulator(pSystem->num_states(), pSystem->num_inputs(),
                         pSystem->num_lambdas(), pSystem->N(), pSystem->dt());
  EXPECT_EQ(simulator.get_input_port_state().size(), pSystem->num_states());
  EXPECT_EQ(simulator.get_input_port_action().size(), pSystem->num_inputs());
  EXPECT_EQ(simulator.get_output_port_next_state().size(),
            pSystem->num_states());
}

TEST_F(LCSSimulatorTest, SimulateOneStep) {
  // Simulate one step and check the output.
  simulator_.CalcOutput(*context_, output_.get());
  const auto& next_state = output_->get_vector_data(0)->get_value();
  EXPECT_EQ(next_state.size(), pSystem->num_states());
}

TEST_F(LCSSimulatorTest, MissingLCS) {
  // Should throw if LCS input is missing
  auto new_context = simulator_.CreateDefaultContext();
  simulator_.get_input_port(1).FixValue(new_context.get(), x0);
  VectorX<double> ones = VectorX<double>::Ones(pSystem->num_inputs());
  simulator_.get_input_port(2).FixValue(new_context.get(), ones);

  EXPECT_THROW(simulator_.CalcOutput(*new_context, output_.get()),
               std::runtime_error);
}

// Test fixture for C3Controller
template <typename T>
class C3ControllerTypedTest : public ::testing::Test, public C3CartpoleProblem {
 protected:
  C3ControllerTypedTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81, 10,
                          0.1) {
    // Load controller options from YAML
    if (std::is_same<T, C3Plus>::value) {
      controller_options_ = drake::yaml::LoadYamlFile<C3ControllerOptions>(
          "examples/resources/cartpole_softwalls/"
          "c3Plus_controller_cartpole_options.yaml");
      UseC3Plus();
    } else
      controller_options_ = drake::yaml::LoadYamlFile<C3ControllerOptions>(
          "examples/resources/cartpole_softwalls/"
          "c3_controller_cartpole_options.yaml");
  }

  void SetUp() override {
    controller_options_.publish_frequency = 0;  // Forced Update

    // Add prediction for the slider joint
    controller_options_.state_prediction_joints.push_back(
        {.name = "CartSlider", .max_acceleration = 10.0});
    controller_options_.lcs_factory_options.N = 10;
    controller_options_.lcs_factory_options.dt = 0.1;

    // Build plant and scene graph
    std::tie(plant, scene_graph) =
        AddMultibodyPlantSceneGraph(&plant_builder_, 0.0);
    Parser parser(plant, scene_graph);
    parser.AddModels(
        "examples/resources/cartpole_softwalls/cartpole_softwalls.sdf");
    plant->Finalize();

    controller_ =
        std::make_unique<C3Controller>(*plant, cost, controller_options_);

    context_ = controller_->CreateDefaultContext();
    output_ = controller_->AllocateOutput();
    discrete_values_ = controller_->AllocateDiscreteVariables();

    // Set up dummy inputs: state, LCS, and target
    auto state_vec =
        c3::systems::TimestampedVector<double>(pSystem->num_states());
    state_vec.SetDataVector(x0);
    state_vec.set_timestamp(0.0);
    controller_->get_input_port_lcs_state().FixValue(context_.get(), state_vec);
    controller_->get_input_port_lcs().FixValue(context_.get(), *pSystem);
    controller_->get_input_port_target().FixValue(context_.get(),
                                                  xdesired.at(0));
  }

  MultibodyPlant<double>* plant{};
  SceneGraph<double>* scene_graph{};
  DiagramBuilder<double> plant_builder_;
  std::unique_ptr<C3Controller> controller_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
  std::unique_ptr<DiscreteValues<double>> discrete_values_;
  C3ControllerOptions controller_options_;
};

using projection_types = ::testing::Types<C3MIQP, C3Plus>;
TYPED_TEST_SUITE(C3ControllerTypedTest, projection_types);

TYPED_TEST(C3ControllerTypedTest, CheckInputOutputPorts) {
  auto& controller_ = this->controller_;
  auto& pSystem = this->pSystem;
  // Check input and output port sizes and existence
  EXPECT_NO_THROW(controller_->get_input_port_lcs_state());
  EXPECT_EQ(controller_->get_input_port_lcs_state().size(),
            pSystem->num_states() + 1);  // +1 for timestamp
  EXPECT_NO_THROW(controller_->get_input_port_lcs());
  EXPECT_EQ(controller_->get_input_port_target().size(), pSystem->num_states());
  EXPECT_NO_THROW(controller_->get_output_port_c3_solution());
  EXPECT_NO_THROW(controller_->get_output_port_c3_intermediates());
}

TYPED_TEST(C3ControllerTypedTest, CheckPlannedTrajectory) {
  auto& controller_ = this->controller_;
  auto& context_ = this->context_;
  auto& discrete_values_ = this->discrete_values_;
  auto& output_ = this->output_;
  auto& pSystem = this->pSystem;
  // Should not throw when computing plan with valid inputs
  EXPECT_NO_THROW({
    controller_->CalcForcedDiscreteVariableUpdate(*context_,
                                                  discrete_values_.get());
    controller_->CalcOutput(*context_, output_.get());
  });

  // Check C3 solution output
  const auto& c3_solution =
      output_->get_data(0)->template get_value<C3Output::C3Solution>();
  EXPECT_EQ(c3_solution.time_vector_.size(), pSystem->N());
  EXPECT_EQ(c3_solution.x_sol_.rows(), pSystem->num_states());
  EXPECT_EQ(c3_solution.lambda_sol_.rows(), pSystem->num_lambdas());
  EXPECT_EQ(c3_solution.u_sol_.rows(), pSystem->num_inputs());
  EXPECT_EQ(c3_solution.x_sol_.cols(), pSystem->N());
  EXPECT_EQ(c3_solution.lambda_sol_.cols(), pSystem->N());
  EXPECT_EQ(c3_solution.u_sol_.cols(), pSystem->N());

  // Check C3 intermediates output
  const auto& c3_intermediates =
      output_->get_data(1)->template get_value<C3Output::C3Intermediates>();
  EXPECT_EQ(c3_intermediates.time_vector_.size(), pSystem->N());
  int total_vars =
      pSystem->num_states() + pSystem->num_lambdas() + pSystem->num_inputs();
  if constexpr (std::is_same<TypeParam, C3Plus>::value) {
    // C3Plus has additional slack variables
    total_vars += pSystem->num_lambdas();
  }
  EXPECT_EQ(c3_intermediates.z_.rows(), total_vars);
  EXPECT_EQ(c3_intermediates.delta_.rows(), total_vars);
  EXPECT_EQ(c3_intermediates.w_.rows(), total_vars);
  EXPECT_EQ(c3_intermediates.z_.cols(), pSystem->N());
  EXPECT_EQ(c3_intermediates.delta_.cols(), pSystem->N());
  EXPECT_EQ(c3_intermediates.w_.cols(), pSystem->N());
}

TYPED_TEST(C3ControllerTypedTest, ThrowsOnMissingLCS) {
  auto& controller_ = this->controller_;
  auto& plant = this->plant;
  auto& x0 = this->x0;
  auto& discrete_values_ = this->discrete_values_;
  // Remove LCS input and expect a runtime error
  auto new_context = controller_->CreateDefaultContext();
  auto state_vec = c3::systems::TimestampedVector<double>(
      plant->num_positions() + plant->num_velocities());
  state_vec.SetDataVector(x0);
  state_vec.set_timestamp(0.0);
  controller_->get_input_port_lcs_state().FixValue(new_context.get(),
                                                   state_vec);
  controller_->get_input_port_target().FixValue(new_context.get(), x0);

  EXPECT_THROW(controller_->CalcForcedDiscreteVariableUpdate(
                   *new_context, discrete_values_.get()),
               std::runtime_error);
}

TYPED_TEST(C3ControllerTypedTest, TestJointPrediction) {
  auto& controller_ = this->controller_;
  auto& context_ = this->context_;
  auto& discrete_values_ = this->discrete_values_;
  auto& output_ = this->output_;
  auto& x0 = this->x0;
  // Test that joint prediction modifies the first state in the planned
  // trajectory
  double eps = 1e-8;
  controller_->CalcForcedDiscreteVariableUpdate(*context_,
                                                discrete_values_.get());
  controller_->CalcOutput(*context_, output_.get());
  const auto& pre_solution =
      output_->get_data(0)->template get_value<C3Output::C3Solution>();
  Eigen::VectorXd first_state =
      pre_solution.x_sol_.col(0).template cast<double>();
  EXPECT_TRUE(first_state.isApprox(x0, eps));

  // Update context with predicted state and check that it changes
  context_->SetDiscreteState(*discrete_values_);
  EXPECT_NO_THROW(controller_->CalcForcedDiscreteVariableUpdate(
      *context_, discrete_values_.get()));
  controller_->CalcOutput(*context_, output_.get());
  const auto& post_solution =
      output_->get_data(0)->template get_value<C3Output::C3Solution>();
  Eigen::VectorXd predicted_first_state =
      post_solution.x_sol_.col(0).template cast<double>();
  EXPECT_FALSE(predicted_first_state.isApprox(x0, eps));
}

// Test fixture for LCSFactorySystem
class LCSFactorySystemTest : public ::testing::Test, public C3CartpoleProblem {
 protected:
  void SetUp() override {
    // Build plant and scene graph
    std::tie(plant, scene_graph) = AddMultibodyPlantSceneGraph(&builder, 0.0);
    Parser parser(plant, scene_graph);
    parser.AddModels(
        "examples/resources/cartpole_softwalls/cartpole_softwalls.sdf");
    plant->Finalize();

    plant_ad = drake::systems::System<double>::ToAutoDiffXd(*plant);
    diagram = builder.Build();

    diagram_context = diagram->CreateDefaultContext();
    context =
        &diagram->GetMutableSubsystemContext(*plant, diagram_context.get());
    context_ad = plant_ad->CreateDefaultContext();

    // Set up contact geometry pairs
    auto left_wall_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("left_wall"));
    auto right_wall_geoms = plant->GetCollisionGeometriesForBody(
        plant->GetBodyByName("right_wall"));
    auto pole_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("Pole"));
    contact_pairs.emplace_back(left_wall_geoms[0], pole_geoms[0]);
    contact_pairs.emplace_back(right_wall_geoms[0], pole_geoms[0]);

    // Load controller options
    controller_options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
        "examples/resources/cartpole_softwalls/"
        "c3_controller_cartpole_options.yaml");

    // Construct LCSFactorySystem
    lcs_factory_system = std::make_unique<c3::systems::LCSFactorySystem>(
        *plant, *context, *plant_ad, *context_ad, contact_pairs,
        controller_options.lcs_factory_options);

    lcs_context = lcs_factory_system->CreateDefaultContext();
    lcs_output = lcs_factory_system->AllocateOutput();

    // Set up dummy state and input
    auto state_vec = c3::systems::TimestampedVector<double>(
        plant->num_positions() + plant->num_velocities());
    state_vec.SetDataVector(x0);
    state_vec.set_timestamp(0.0);
    lcs_factory_system->get_input_port_lcs_state().FixValue(lcs_context.get(),
                                                            state_vec);

    Eigen::VectorXd u = Eigen::VectorXd::Zero(plant->num_actuators());
    lcs_factory_system->get_input_port_lcs_input().FixValue(lcs_context.get(),
                                                            u);
  }

  drake::systems::DiagramBuilder<double> builder;
  std::unique_ptr<drake::systems::Diagram<double>> diagram;
  std::unique_ptr<drake::systems::Context<double>> diagram_context;
  MultibodyPlant<double>* plant{};
  SceneGraph<double>* scene_graph{};
  std::unique_ptr<drake::multibody::MultibodyPlant<drake::AutoDiffXd>> plant_ad;
  drake::systems::Context<double>* context{};
  std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>> context_ad;
  std::unique_ptr<c3::systems::LCSFactorySystem> lcs_factory_system;
  std::unique_ptr<drake::systems::Context<double>> lcs_context;
  std::unique_ptr<drake::systems::SystemOutput<double>> lcs_output;
  C3ControllerOptions controller_options;
  std::vector<SortedPair<GeometryId>> contact_pairs;
};

TEST_F(LCSFactorySystemTest, InputOutputPortSizes) {
  // Check input port sizes and output port existence
  EXPECT_EQ(lcs_factory_system->get_input_port_lcs_state().size(),
            plant->num_positions() + plant->num_velocities() +
                1);  // +1 for timestamp
  EXPECT_EQ(lcs_factory_system->get_input_port_lcs_input().size(),
            plant->num_actuators());
  EXPECT_NO_THROW(lcs_factory_system->get_output_port_lcs());
  EXPECT_NO_THROW(
      lcs_factory_system->get_output_port_lcs_contact_descriptions());
}

TEST_F(LCSFactorySystemTest, OutputLCSIsValid) {
  // Should not throw and should produce an LCS object with correct dimensions
  EXPECT_NO_THROW(
      { lcs_factory_system->CalcOutput(*lcs_context, lcs_output.get()); });
  const auto& lcs = lcs_output->get_data(0)->get_value<c3::LCS>();
  EXPECT_EQ(lcs.num_states(), plant->num_positions() + plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(), LCSFactory::GetNumContactVariables(
                                   controller_options.lcs_factory_options));
  EXPECT_EQ(lcs.dt(), controller_options.lcs_factory_options.dt);
  EXPECT_EQ(lcs.N(), controller_options.lcs_factory_options.N);
}

TEST_F(LCSFactorySystemTest, OutputContactJacobianIsValid) {
  // Check that the contact Jacobian output is valid
  EXPECT_NO_THROW(
      { lcs_factory_system->CalcOutput(*lcs_context, lcs_output.get()); });
  auto contact_descriptions =
      lcs_output->get_data(1)->get_value<std::vector<LCSContactDescription>>();
  EXPECT_EQ(contact_descriptions.size(),
            contact_pairs.size());  // for two pairs of contacts
  EXPECT_EQ(contact_descriptions.back().witness_point_A.size(),
            3);  // 3D coordinate point
  EXPECT_FALSE(
      contact_descriptions.back().witness_point_A.isZero());  // Not zero
  EXPECT_EQ(contact_descriptions.back().witness_point_B.size(),
            3);  // 3D coordinate point
  EXPECT_FALSE(
      contact_descriptions.back().witness_point_B.isZero());  // Not zero
  EXPECT_EQ(contact_descriptions.back().force_basis.size(),
            3);  // 3D force basis
  EXPECT_FALSE(contact_descriptions.back().force_basis.isZero());  // Not zero
}

}  // namespace test
}  // namespace systems
}  // namespace c3
