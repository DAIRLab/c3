#include <gtest/gtest.h>

#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"
#include "multibody/multibody_utils.h"

#include "drake/common/sorted_pair.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"

/**
 * @file multibody_test.cc
 * @brief Tests for multibody dynamics components including LCSFactory and
 * GeomGeomCollider.
 *
 * The tests cover:
 *  - LCSFactory: Generating Linear Complementarity Systems (LCS) for multibody
 *    systems with contact, including:
 *    - Calculating the number of contact variables for different contact models
 *    - Generating and linearizing an LCS from a MultibodyPlant
 *    - Updating state and input
 *    - Computing contact Jacobians
 *    - Fixing modes in an LCS
 *  - GeomGeomCollider: Computing distances and Jacobians between geometry pairs
 *
 * The tests use a cube pivoting example and sphere-mesh collision scenarios
 * to verify correctness. Different contact models (Stewart-Trinkle, Anitescu,
 * and Frictionless Spring) are tested.
 */
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

// Test the static method for computing number of contact variables
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
  EXPECT_THROW(LCSFactory::GetNumContactVariables(ContactModel::kUnknown, 3, 0),
               std::out_of_range);

  // Test with LCSFactoryOptions struct
  LCSFactoryOptions options;
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

  // Test error handling for invalid contact model
  options.contact_model = "some_random_contact_model";
  EXPECT_THROW(LCSFactory::GetNumContactVariables(options), std::out_of_range);
}

// Helper struct to hold common test fixture data for cube pivoting scenarios
struct PivotingTestFixture {
  DiagramBuilder<double> plant_builder;
  MultibodyPlant<double>* plant;
  SceneGraph<double>* scene_graph;
  std::unique_ptr<MultibodyPlant<AutoDiffXd>> plant_autodiff;
  std::unique_ptr<Context<AutoDiffXd>> plant_context_autodiff;
  std::unique_ptr<Diagram<double>> plant_diagram;
  std::unique_ptr<Context<double>> plant_diagram_context;
  std::vector<SortedPair<GeometryId>> contact_pairs;
  LCSFactoryOptions options;
  std::unique_ptr<LCSFactory> lcs_factory;
  ContactModel contact_model = ContactModel::kUnknown;

  // Initialize the multibody plant and LCS factory for cube pivoting tests
  void Initialize() {
    // Create plant and scene graph
    std::tie(plant, scene_graph) =
        AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
    Parser parser(plant, scene_graph);
    parser.AddModels("examples/resources/cube_pivoting/cube_pivoting.sdf");
    plant->Finalize();

    // Load LCS factory options from YAML configuration
    options = drake::yaml::LoadYamlFile<LCSFactoryOptions>(
        "multibody/test/resources/lcs_factory_pivoting_options.yaml");

    // Initialize friction directions per contact if not set in config
    if (!options.num_friction_directions_per_contact) {
      options.num_friction_directions_per_contact = std::vector<int>(
          options.num_contacts, options.num_friction_directions.value());
    }

    // Build diagram and create contexts
    plant_diagram = plant_builder.Build();
    plant_diagram_context = plant_diagram->CreateDefaultContext();

    auto& plant_context = plant_diagram->GetMutableSubsystemContext(
        *plant, plant_diagram_context.get());

    // Create autodiff version for automatic differentiation
    plant_autodiff = drake::systems::System<double>::ToAutoDiffXd(*plant);
    plant_context_autodiff = plant_autodiff->CreateDefaultContext();

    // Retrieve collision geometries for all bodies
    auto platform_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("platform"));
    auto cube_geoms =
        plant->GetCollisionGeometriesForBody(plant->GetBodyByName("cube"));
    auto left_finger_geoms = plant->GetCollisionGeometriesForBody(
        plant->GetBodyByName("left_finger"));
    auto right_finger_geoms = plant->GetCollisionGeometriesForBody(
        plant->GetBodyByName("right_finger"));

    // Define contact pairs between cube and other bodies
    contact_pairs.emplace_back(cube_geoms[0], left_finger_geoms[0]);
    contact_pairs.emplace_back(cube_geoms[0], platform_geoms[0]);
    contact_pairs.emplace_back(cube_geoms[0], right_finger_geoms[0]);

    // Initialize state and input vectors to zero
    drake::VectorX<double> state =
        VectorXd::Zero(plant->num_positions() + plant->num_velocities());
    drake::VectorX<double> input = VectorXd::Zero(plant->num_actuators());

    // Create LCS factory with contact pairs
    lcs_factory = std::make_unique<LCSFactory>(
        *plant, plant_context, *plant_autodiff, *plant_context_autodiff,
        contact_pairs, options);
    lcs_factory->UpdateStateAndInput(state, input);
  }
};

// Test fixture for non-parameterized LCS factory tests
class LCSFactoryPivotingTest : public testing::Test {
 protected:
  void SetUp() override { fixture.Initialize(); }

  PivotingTestFixture fixture;
};

// Test that contact pairs can be parsed from options instead of explicit list
TEST_F(LCSFactoryPivotingTest, ContactPairParsing) {
  std::unique_ptr<LCSFactory> contact_pair_parsed_factory;

  // Create factory without explicit contact pairs (parsed from options)
  EXPECT_NO_THROW({
    contact_pair_parsed_factory = std::make_unique<LCSFactory>(
        *fixture.plant,
        fixture.plant_diagram->GetMutableSubsystemContext(
            *fixture.plant, fixture.plant_diagram_context.get()),
        *fixture.plant_autodiff, *fixture.plant_context_autodiff,
        fixture.options);
  });
  EXPECT_NE(contact_pair_parsed_factory.get(), nullptr);

  // Update factory with zero state and input
  drake::VectorX<double> state = VectorXd::Zero(
      fixture.plant->num_positions() + fixture.plant->num_velocities());
  drake::VectorX<double> input = VectorXd::Zero(fixture.plant->num_actuators());
  contact_pair_parsed_factory->UpdateStateAndInput(state, input);

  // Generate LCS and verify dimensions
  LCS lcs = contact_pair_parsed_factory->GenerateLCS();

  EXPECT_EQ(lcs.num_states(),
            fixture.plant->num_positions() + fixture.plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), fixture.plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(),
            LCSFactory::GetNumContactVariables(fixture.options));
}

// Parameterized test fixture for testing different contact models and friction
// directions
class LCSFactoryParameterizedPivotingTest
    : public ::testing::TestWithParam<std::tuple<std::string, int>> {
 protected:
  void SetUp() override {
    fixture.Initialize();

    // Initialize state and input to zero
    drake::VectorX<double> state = VectorXd::Zero(
        fixture.plant->num_positions() + fixture.plant->num_velocities());
    drake::VectorX<double> input =
        VectorXd::Zero(fixture.plant->num_actuators());

    // Set contact model and friction directions from test parameters
    fixture.options.contact_model = std::get<0>(GetParam());
    fixture.contact_model =
        GetContactModelMap().at(fixture.options.contact_model);
    fixture.options.num_friction_directions_per_contact =
        std::vector<int>(fixture.contact_pairs.size(), std::get<1>(GetParam()));

    // Create factory with parameterized options
    fixture.lcs_factory = std::make_unique<LCSFactory>(
        *fixture.plant,
        fixture.plant_diagram->GetMutableSubsystemContext(
            *fixture.plant, fixture.plant_diagram_context.get()),
        *fixture.plant_autodiff, *fixture.plant_context_autodiff,
        fixture.contact_pairs, fixture.options);
    fixture.lcs_factory->UpdateStateAndInput(state, input);
  }

  PivotingTestFixture fixture;
};

// Test LCS generation for different contact models
TEST_P(LCSFactoryParameterizedPivotingTest, GenerateLCS) {
  LCS lcs = fixture.lcs_factory->GenerateLCS();

  // Verify LCS dimensions match plant configuration
  EXPECT_EQ(lcs.num_states(),
            fixture.plant->num_positions() + fixture.plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), fixture.plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(),
            LCSFactory::GetNumContactVariables(fixture.options));
}

// Test static linearization method for different contact models
TEST_P(LCSFactoryParameterizedPivotingTest, LinearizePlantToLCS) {
  auto& plant_context = fixture.plant_diagram->GetMutableSubsystemContext(
      *fixture.plant, fixture.plant_diagram_context.get());

  drake::VectorX<double> state = VectorXd::Zero(
      fixture.plant->num_positions() + fixture.plant->num_velocities());
  drake::VectorX<double> input = VectorXd::Zero(fixture.plant->num_actuators());

  // Use static method to linearize plant directly to LCS
  LCS lcs = LCSFactory::LinearizePlantToLCS(
      *fixture.plant, plant_context, *fixture.plant_autodiff,
      *fixture.plant_context_autodiff, fixture.contact_pairs, fixture.options,
      state, input);

  // Verify linearized LCS dimensions
  EXPECT_EQ(lcs.num_states(),
            fixture.plant->num_positions() + fixture.plant->num_velocities());
  EXPECT_EQ(lcs.num_inputs(), fixture.plant->num_actuators());
  EXPECT_EQ(lcs.num_lambdas(),
            LCSFactory::GetNumContactVariables(fixture.options));
}

// Test that updating state and input changes contact-dependent LCS matrices
TEST_P(LCSFactoryParameterizedPivotingTest, UpdateStateAndInput) {
  // Generate initial LCS at zero state
  LCS initial_lcs = fixture.lcs_factory->GenerateLCS();
  auto initial_contact_descriptions =
      fixture.lcs_factory->GetContactDescriptions();

  // Update to non-zero state (cube tilted and positioned above platform)
  drake::VectorX<double> state = VectorXd::Zero(
      fixture.plant->num_positions() + fixture.plant->num_velocities());
  state << 0, 0.75, 0.785, -0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0;
  drake::VectorX<double> input = VectorXd::Zero(fixture.plant->num_actuators());
  input << 0, 0, 0, 0;

  fixture.lcs_factory->UpdateStateAndInput(state, input);

  // Generate updated LCS
  LCS updated_lcs = fixture.lcs_factory->GenerateLCS();
  auto updated_contact_descriptions =
      fixture.lcs_factory->GetContactDescriptions();

  EXPECT_EQ(initial_lcs.A(), updated_lcs.A());
  EXPECT_EQ(initial_lcs.B(), updated_lcs.B());
  EXPECT_EQ(initial_lcs.d(), updated_lcs.d());

  // Contact-dependent matrices should change due to different configuration
  EXPECT_NE(initial_lcs.D(), updated_lcs.D());
  if (fixture.contact_model != ContactModel::kFrictionlessSpring) {
    EXPECT_NE(initial_lcs.E(), updated_lcs.E());
    EXPECT_NE(initial_lcs.F(), updated_lcs.F());
    EXPECT_NE(initial_lcs.H(), updated_lcs.H());
    EXPECT_NE(initial_lcs.c(), updated_lcs.c());
  }

  // Contact Jacobian and points should change
  for (size_t i = 0; i < initial_contact_descriptions.size(); ++i) {
    if (initial_contact_descriptions[i].is_slack) continue;
    EXPECT_NE(initial_contact_descriptions[i].witness_point_A,
              updated_contact_descriptions[i].witness_point_A);
    EXPECT_NE(initial_contact_descriptions[i].witness_point_B,
              updated_contact_descriptions[i].witness_point_B);
    EXPECT_NE(initial_contact_descriptions[i].force_basis,
              updated_contact_descriptions[i].force_basis);
  }
}

// Test contact Jacobian computation for different contact models
TEST_P(LCSFactoryParameterizedPivotingTest, CheckContactDescriptionSizes) {
  auto contact_descriptions = fixture.lcs_factory->GetContactDescriptions();

  int n_contacts = fixture.contact_pairs.size();
  auto n_tangential_directions =
      2 * std::accumulate(
              fixture.options.num_friction_directions_per_contact->begin(),
              fixture.options.num_friction_directions_per_contact->end(), 0);

  for (size_t i = 0; i < contact_descriptions.size(); ++i) {
    if (contact_descriptions[i].is_slack) continue;
    EXPECT_FALSE(contact_descriptions[i].witness_point_A.isZero());
    EXPECT_FALSE(contact_descriptions[i].witness_point_B.isZero());
    EXPECT_FALSE(contact_descriptions[i].force_basis.isZero());
  }

  // Verify Jacobian rows based on contact model
  switch (fixture.contact_model) {
    case ContactModel::kStewartAndTrinkle:
      // Normal + tangential directions for all contacts
      EXPECT_EQ(contact_descriptions.size(),
                2*n_contacts + n_tangential_directions);
      break;
    case ContactModel::kFrictionlessSpring:
      // Normal directions only
      EXPECT_EQ(contact_descriptions.size(), n_contacts);
      break;
    case ContactModel::kAnitescu:
      // Tangential directions only (normal handled differently)
      EXPECT_EQ(contact_descriptions.size(), n_tangential_directions);
      break;
    default:
      EXPECT_TRUE(false);
  }
}

// Test fixing specific contact modes in the LCS
TEST_P(LCSFactoryParameterizedPivotingTest, FixSomeModes) {
  LCS lcs = fixture.lcs_factory->GenerateLCS();

  // Fix modes at indices 0 and 1 (removes 2 lambda variables)
  LCS new_lcs = LCSFactory::FixSomeModes(lcs, {0}, {1});
  int updated_num_lambda = lcs.num_lambdas() - 2;

  // Verify all contact-related matrices have reduced dimensions
  EXPECT_EQ(new_lcs.D()[0].cols(), updated_num_lambda);
  EXPECT_EQ(new_lcs.E()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.F()[0].cols(), updated_num_lambda);
  EXPECT_EQ(new_lcs.F()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.H()[0].rows(), updated_num_lambda);
  EXPECT_EQ(new_lcs.c()[0].rows(), updated_num_lambda);
}

// Instantiate parameterized tests with different contact models and friction
// directions
INSTANTIATE_TEST_SUITE_P(ContactModelTests, LCSFactoryParameterizedPivotingTest,
                         ::testing::Values(std::tuple("frictionless_spring", 0),
                                           std::tuple("stewart_and_trinkle", 1),
                                           std::tuple("stewart_and_trinkle", 2),
                                           std::tuple("anitescu", 1),
                                           std::tuple("anitescu", 2)));

// Test distance computation between sphere and mesh geometries
GTEST_TEST(GeomGeomColliderTest, SphereMeshDistance) {
  auto MESH_HEIGHT = 0.015;  // Approximate height of mesh above z=0 plane
  auto SPHERE_RADIUS = 0.01;
  auto NO_CONTACT_HEIGHT = 0.1;  // Height where sphere is not in contact

  // Setup plant with sphere and C-shaped mesh
  DiagramBuilder<double> plant_builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&plant_builder, 0.0);
  Parser parser(&plant, &scene_graph);
  parser.AddModels("multibody/test/resources/sphere-and-mesh.sdf");

  // Position sphere well above mesh (no contact)
  plant.SetDefaultFreeBodyPose(
      plant.GetBodyByName("sphere"),
      drake::math::RigidTransformd(
          Eigen::Quaterniond::Identity(),
          Eigen::Vector3d(0.0, 0.0, NO_CONTACT_HEIGHT)));
  plant.Finalize();

  auto diagram = plant_builder.Build();
  auto diagram_context = diagram->CreateDefaultContext();
  auto& context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Get collision geometries for sphere and mesh
  auto sphere_geoms =
      plant.GetCollisionGeometriesForBody(plant.GetBodyByName("sphere"));
  auto mesh_geoms =
      plant.GetCollisionGeometriesForBody(plant.GetBodyByName("mesh"));

  ASSERT_FALSE(sphere_geoms.empty());
  ASSERT_FALSE(mesh_geoms.empty());

  auto geom_pair = SortedPair<GeometryId>(sphere_geoms[0], mesh_geoms[0]);

  // Test sphere not in contact (positive distance)
  GeomGeomCollider<double> collider(plant, geom_pair);
  auto [distance_no_contact, J] = collider.EvalPolytope(context, 1);

  EXPECT_NEAR(distance_no_contact,
              NO_CONTACT_HEIGHT - MESH_HEIGHT - SPHERE_RADIUS,
              1e-2);  // Expected distance
  EXPECT_EQ(J.rows(), 3);
  EXPECT_EQ(J.cols(), plant.num_velocities());
  EXPECT_TRUE(J.allFinite());

  // Test sphere in contact (sphere center at radius height)
  plant.SetFreeBodyPose(
      &context, plant.GetBodyByName("sphere"),
      drake::math::RigidTransformd(Eigen::Quaterniond::Identity(),
                                   Eigen::Vector3d(0.0, 0.0, SPHERE_RADIUS)));

  GeomGeomCollider<double> collider_contact(plant, geom_pair);
  auto [distance_contact, J_contact] =
      collider_contact.EvalPolytope(context, 1);

  EXPECT_NEAR(distance_contact, -MESH_HEIGHT,
              1e-2);  // Expect penetration or zero distance
  EXPECT_EQ(J_contact.rows(), 3);
  EXPECT_EQ(J_contact.cols(), plant.num_velocities());
  EXPECT_TRUE(J_contact.allFinite());
}

// Parameterized test for GetNClosestContactPairs with different values of N
class GetNClosestContactPairsTest
    : public ::testing::TestWithParam<unsigned int> {
 protected:
  void SetUp() override {
    // Setup plant with scene graph for collision detection
    DiagramBuilder<double> builder;
    std::tie(plant_ptr, scene_graph) =
        AddMultibodyPlantSceneGraph(&builder, 0.0);

    // Use fixed seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> height_dist(0.05, 0.5);

    // Add 5 cubes at random heights
    for (int i = 0; i < 5; ++i) {
      std::string cube_name = "cube_" + std::to_string(i);
      object_names.push_back(cube_name);

      double height = height_dist(rng);

      // Create cube shape (0.1m x 0.1m x 0.1m)
      auto cube_shape = drake::geometry::Box(0.1, 0.1, 0.1);
      const auto& cube_body = plant_ptr->AddRigidBody(
          cube_name, drake::multibody::SpatialInertia<double>::MakeUnitary());

      // Position cube at height above ground
      drake::math::RigidTransformd cube_pose(
          Eigen::Quaterniond::Identity(),
          Eigen::Vector3d(i * 0.2, 0.0, height));

      plant_ptr->RegisterCollisionGeometry(
          cube_body, drake::math::RigidTransformd(), cube_shape,
          cube_name + "_collision",
          drake::multibody::CoulombFriction<double>(0.5, 0.5));

      plant_ptr->SetDefaultFreeBodyPose(cube_body, cube_pose);

      // Store height for this object
      object_ground_distances[cube_name] =
          height - 0.05;  // Distance from ground is height minus half cube size
    }

    // Add 5 spheres at random heights
    for (int i = 0; i < 5; ++i) {
      std::string sphere_name = "sphere_" + std::to_string(i);
      object_names.push_back(sphere_name);

      double height = height_dist(rng);

      // Create sphere shape (radius = 0.05m)
      auto sphere_shape = drake::geometry::Sphere(0.05);
      const auto& sphere_body = plant_ptr->AddRigidBody(
          sphere_name, drake::multibody::SpatialInertia<double>::MakeUnitary());

      // Position sphere at height above ground (offset in y direction)
      drake::math::RigidTransformd sphere_pose(
          Eigen::Quaterniond::Identity(),
          Eigen::Vector3d(i * 0.2, 0.3, height));

      plant_ptr->RegisterCollisionGeometry(
          sphere_body, drake::math::RigidTransformd(), sphere_shape,
          sphere_name + "_collision",
          drake::multibody::CoulombFriction<double>(0.5, 0.5));

      plant_ptr->SetDefaultFreeBodyPose(sphere_body, sphere_pose);

      // Store height for this object
      object_ground_distances[sphere_name] =
          height - 0.05;  // Distance from ground is height minus radius
    }

    // Add ground plane (table)
    double table_thickness = 0.05;
    double table_width = 2.0;
    auto table_shape =
        drake::geometry::Box(table_width, table_width, table_thickness);
    const auto& table_body = plant_ptr->AddRigidBody(
        "table", drake::multibody::SpatialInertia<double>::MakeUnitary());

    drake::math::RigidTransformd table_pose(
        Eigen::Quaterniond::Identity(),
        Eigen::Vector3d(0.0, 0.0, -table_thickness / 2.0));

    plant_ptr->RegisterCollisionGeometry(
        table_body, drake::math::RigidTransformd(), table_shape,
        "table_collision", drake::multibody::CoulombFriction<double>(1.0, 1.0));

    plant_ptr->WeldFrames(plant_ptr->world_frame(), table_body.body_frame(),
                          table_pose);

    plant_ptr->Finalize();

    // Build diagram and get context
    diagram = builder.Build();
    diagram_context = diagram->CreateDefaultContext();
    context =
        &diagram->GetMutableSubsystemContext(*plant_ptr, diagram_context.get());

    // Create contact pairs between all objects and table
    auto table_geoms = plant_ptr->GetCollisionGeometriesForBody(
        plant_ptr->GetBodyByName("table"));

    for (const auto& object_name : object_names) {
      auto object_geoms = plant_ptr->GetCollisionGeometriesForBody(
          plant_ptr->GetBodyByName(object_name));

      if (!object_geoms.empty() && !table_geoms.empty()) {
        auto pair = SortedPair<GeometryId>(object_geoms[0], table_geoms[0]);
        all_pairs.push_back(pair);

        // Map contact pair to object height (proxy for distance)
        pair_to_distance[pair] = object_ground_distances[object_name];
      }
    }

    ASSERT_EQ(all_pairs.size(), 10);  // 5 cubes + 5 spheres
  }

  MultibodyPlant<double>* plant_ptr;
  SceneGraph<double>* scene_graph;
  std::unique_ptr<Diagram<double>> diagram;
  std::unique_ptr<Context<double>> diagram_context;
  Context<double>* context;

  std::vector<std::string> object_names;
  std::map<std::string, double> object_ground_distances;
  std::vector<SortedPair<GeometryId>> all_pairs;
  std::map<SortedPair<GeometryId>, double> pair_to_distance;
};

TEST_P(GetNClosestContactPairsTest, SelectsClosestN) {
  unsigned int N = GetParam();

  // Handle case where N exceeds available pairs
  if (N > all_pairs.size()) {
    // DRAKE_DEMAND aborts the process, not throws an exception
    EXPECT_DEATH(
        LCSFactory::GetNClosestContactPairs(*plant_ptr, *context, all_pairs, N),
        "condition 'N <= contact_pairs.size\\(\\)' failed");
    return;
  }

  // Get N closest contact pairs
  auto selected =
      LCSFactory::GetNClosestContactPairs(*plant_ptr, *context, all_pairs, N);
  ASSERT_EQ(selected.size(), N);

  if (N == 0) {
    // If N=0, no pairs should be selected
    EXPECT_TRUE(selected.empty());
    return;
  }

  // Get distances for selected pairs
  std::vector<double> selected_distances;
  selected_distances.reserve(selected.size());
  for (const auto& pair : selected) {
    ASSERT_TRUE(pair_to_distance.count(pair) > 0);
    selected_distances.push_back(pair_to_distance.at(pair));
  }

  // Sort selected distances
  std::sort(selected_distances.begin(), selected_distances.end());

  // Get all distances and sort them
  std::vector<double> all_distances;
  all_distances.reserve(all_pairs.size());
  for (const auto& [pair, distance] : pair_to_distance) {
    all_distances.push_back(distance);
  }
  std::sort(all_distances.begin(), all_distances.end());

  // Verify selected distances match the N smallest
  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(selected_distances[i], all_distances[i], 1e-6)
        << "Selected distance at index " << i << " is " << selected_distances[i]
        << " but expected " << all_distances[i];
  }
}

// Test with N = 1, 2, 3, 5, 7, 10 (all), and 15 (exceeds total)
INSTANTIATE_TEST_SUITE_P(DifferentSubsetSizes, GetNClosestContactPairsTest,
                         ::testing::Values(0, 1, 5, 10, 15));

}  // namespace test
}  // namespace multibody
}  // namespace c3
