#include "multibody/lcs_factory.h"

#include <iostream>

#include <drake/multibody/parsing/parser.h>
#include <gtest/gtest.h>

#include "multibody/geom_geom_collider.h"
#include "multibody/multibody_utils.h"

#include "drake/common/sorted_pair.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"

namespace c3 {
namespace multibody {
namespace test {

using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::System;
using Eigen::MatrixXd;
using Eigen::VectorXd;

GTEST_TEST(LCSFactoryTest, GetNumContactVariables) {
  // Test with different contact models and friction properties
  EXPECT_EQ(LCSFactory::GetNumContactVariables(ContactModel::kStewartAndTrinkle,
                                               2, 4),
            20);
  EXPECT_EQ(LCSFactory::GetNumContactVariables(ContactModel::kAnitescu, 3, 2),
            12);
  EXPECT_EQ(LCSFactory::GetNumContactVariables(
                ContactModel::kFrictionlessSpring, 3, 0),
            3);

  // Test with LCSOptions
  LCSOptions options;
  options.contact_model = "stewart_and_trinkle";
  options.num_friction_directions = 4;
  options.num_contacts = 2;
  EXPECT_EQ(LCSFactory::GetNumContactVariables(options), 20);

  options.contact_model = "anitescu";
  options.num_friction_directions = 2;
  options.num_contacts = 3;
  EXPECT_EQ(LCSFactory::GetNumContactVariables(options), 12);

  options.contact_model = "frictionless_spring";
  options.num_friction_directions = 0;
  options.num_contacts = 3;
  EXPECT_EQ(LCSFactory::GetNumContactVariables(options), 3);
}

class LCSFactoryPivotingTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    std::tie(plant, scene_graph) =
        AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
    Parser parser(plant, scene_graph);
    parser.AddModels("systems/test/res/cube_pivoting.sdf");
    plant->Finalize();

    // Load controller options from YAML file.
    options = drake::yaml::LoadYamlFile<C3ControllerOptions>(
        "systems/test/res/c3_pivoting_options.yaml");
    plant_diagram = plant_builder.Build();
    plant_diagram_context = plant_diagram->CreateDefaultContext();

    // Get the context for the plant within the diagram.
    auto& plant_context = plant_diagram->GetMutableSubsystemContext(
        *plant, plant_diagram_context.get());

    // Convert the plant to AutoDiffXd for automatic differentiation.
    plant_autodiff = drake::systems::System<double>::ToAutoDiffXd(*plant);
    plant_context_autodiff = plant_autodiff->CreateDefaultContext();

    // Retrieve collision geometries for relevant bodies.
    std::vector<GeometryId> platform_collision_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("platform"));
    std::vector<GeometryId> cube_collision_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("cube"));
    std::vector<GeometryId> left_finger_collision_geoms =
        plant->GetCollisionGeometriesForBody(
            plant->GetBodyByName("left_finger"));
    std::vector<GeometryId> right_finger_collision_geoms =
        plant->GetCollisionGeometriesForBody(
            plant->GetBodyByName("right_finger"));

    // Map collision geometries to their respective components.
    std::unordered_map<std::string, std::vector<GeometryId>> contact_geoms;
    contact_geoms["PLATFORM"] = platform_collision_geoms;
    contact_geoms["CUBE"] = cube_collision_geoms;
    contact_geoms["LEFT_FINGER"] = left_finger_collision_geoms;
    contact_geoms["RIGHT_FINGER"] = right_finger_collision_geoms;

    // Define contact pairs for the LCS system.
    contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                               contact_geoms["LEFT_FINGER"][0]);
    contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                               contact_geoms["PLATFORM"][0]);
    contact_pairs.emplace_back(contact_geoms["CUBE"][0],
                               contact_geoms["RIGHT_FINGER"][0]);

    // Set initial positions, velocities, and inputs for both double and
    // AutoDiffXd plants.
    drake::VectorX<double> state =
        VectorXd::Zero(plant->num_positions() + plant->num_velocities());
    drake::VectorX<double> input = VectorXd::Zero(plant->num_actuators());

    options.contact_model = GetParam();
    contact_model = ContactModelMap()[GetParam()];
    lcs_factory = std::make_unique<LCSFactory>(
        *plant, plant_context, *plant_autodiff, *plant_context_autodiff,
        contact_pairs, options);
    lcs_factory->UpdateStateAndInput(state, input);
  }

  DiagramBuilder<double> plant_builder;
  MultibodyPlant<double>* plant;
  SceneGraph<double>* scene_graph;
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_autodiff;
  std::unique_ptr<Context<AutoDiffXd>> plant_context_autodiff;
  std::unique_ptr<Diagram<double>> plant_diagram;
  std::unique_ptr<Context<double>> plant_diagram_context;
  std::vector<SortedPair<GeometryId>> contact_pairs;
  LCSOptions options;
  std::unique_ptr<LCSFactory> lcs_factory;
  ContactModel contact_model = ContactModel::kUnknown;
};

TEST_P(LCSFactoryPivotingTest, GenerateLCS) {
  LCS lcs = lcs_factory->GenerateLCS();

  EXPECT_EQ(lcs.num_states(), plant->num_positions() + plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(), LCSFactory::GetNumContactVariables(options));
}

TEST_P(LCSFactoryPivotingTest, LinearizePlantToLCS) {
  // Get the context for the plant within the diagram.
  auto& plant_context = plant_diagram->GetMutableSubsystemContext(
      *plant, plant_diagram_context.get());

  drake::VectorX<double> state =
      VectorXd::Zero(plant->num_positions() + plant->num_velocities());
  drake::VectorX<double> input = VectorXd::Zero(plant->num_actuators());

  LCS lcs = LCSFactory::LinearizePlantToLCS(
      *plant, plant_context, *plant_autodiff, *plant_context_autodiff,
      contact_pairs, options, state, input);

  EXPECT_EQ(lcs.num_states(), plant->num_positions() + plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(), LCSFactory::GetNumContactVariables(options));
}

TEST_P(LCSFactoryPivotingTest, UpdateStateAndInput) {
  LCS initial_lcs = lcs_factory->GenerateLCS();
  auto [initial_J, initial_contact_points] =
      lcs_factory->GetContactJacobianAndPoints();

  drake::VectorX<double> state =
      VectorXd::Zero(plant->num_positions() + plant->num_velocities());
  state << 0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0;
  drake::VectorX<double> input = VectorXd::Zero(plant->num_actuators());
  input << 0, 0, 0, 0;

  // Update the LCS factory with the state and input.
  lcs_factory->UpdateStateAndInput(state, input);

  LCS updated_lcs = lcs_factory->GenerateLCS();
  auto [updated_J, updated_contact_points] =
      lcs_factory->GetContactJacobianAndPoints();

  EXPECT_EQ(initial_lcs.A(), updated_lcs.A());
  EXPECT_EQ(initial_lcs.B(), updated_lcs.B());
  EXPECT_EQ(initial_lcs.d(), updated_lcs.d());

  EXPECT_NE(initial_lcs.D(), updated_lcs.D());
  // Except for frictionless spring, the contact model would generate different
  // matrices
  if (contact_model != ContactModel::kFrictionlessSpring) {
    EXPECT_NE(initial_lcs.E(), updated_lcs.E());
    EXPECT_NE(initial_lcs.F(), updated_lcs.F());
    EXPECT_NE(initial_lcs.H(), updated_lcs.H());
    EXPECT_NE(initial_lcs.c(), updated_lcs.c());
  }

  EXPECT_NE(initial_J, updated_J);
  for (size_t i = 0; i < initial_contact_points.size(); ++i) {
    EXPECT_NE(initial_contact_points[i], updated_contact_points[i]);
  }
}

TEST_P(LCSFactoryPivotingTest, ComputeContactJacobian) {
  auto [J, contact_points] = lcs_factory->GetContactJacobianAndPoints();

  int n_contacts = contact_pairs.size();
  // Check for number of force variables (not including slack variables)
  switch (contact_model) {
    case ContactModel::kStewartAndTrinkle:
      EXPECT_EQ(J.rows(),
                n_contacts + 2 * n_contacts * options.num_friction_directions);
      break;
    case ContactModel::kFrictionlessSpring:
      EXPECT_EQ(J.rows(), n_contacts);
      break;
    case ContactModel::kAnitescu:
      EXPECT_EQ(J.rows(), 2 * n_contacts * options.num_friction_directions);
      break;
    default:
      EXPECT_TRUE(false);  // Something went wrong in parsing the contact model
  }
  EXPECT_EQ(J.cols(), plant->num_velocities());
  EXPECT_EQ(contact_points.size(), n_contacts);
}

TEST_P(LCSFactoryPivotingTest, FixSomeModes) {
  LCS lcs = lcs_factory->GenerateLCS();
  LCS new_lcs = LCSFactory::FixSomeModes(lcs, {0}, {1});
  int updated_num_lambda = lcs.num_lambdas() - 2;  // 1 active, 1 inactive
  EXPECT_EQ(new_lcs.D()[0].cols(), updated_num_lambda);
  EXPECT_EQ(new_lcs.E()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.F()[0].cols(), updated_num_lambda);
  EXPECT_EQ(new_lcs.F()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.H()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.c()[0].rows(), updated_num_lambda);
}

INSTANTIATE_TEST_SUITE_P(ContactModelTests, LCSFactoryPivotingTest,
                         ::testing::Values("frictionless_spring",
                                           "stewart_and_trinkle", "anitescu"));

}  // namespace test
}  // namespace multibody
}  // namespace c3