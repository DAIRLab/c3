// generators_test.cc
// Unit tests for LCM generators in the c3 project.

#include <drake/systems/framework/context.h>
#include <drake/systems/framework/system_output.h>
#include <gtest/gtest.h>

#include "core/test/c3_cartpole_problem.hpp"
#include "multibody/lcs_factory.h"
#include "systems/framework/c3_output.h"
#include "systems/lcmt_generators/c3_output_generator.h"
#include "systems/lcmt_generators/c3_trajectory_generator.h"
#include "systems/lcmt_generators/contact_force_generator.h"

using c3::multibody::LCSContactDescription;
using c3::systems::C3Output;
using c3::systems::lcmt_generators::C3OutputGenerator;
using c3::systems::lcmt_generators::C3TrajectoryGenerator;
using c3::systems::lcmt_generators::C3TrajectoryGeneratorConfig;
using c3::systems::lcmt_generators::ContactForceGenerator;
using c3::systems::lcmt_generators::TrajectoryDescription;
using drake::systems::Context;
using drake::systems::SystemOutput;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace c3 {
namespace systems {
namespace test {

// Test fixture for ContactForceGenerator
class ContactForceGeneratorTest : public ::testing::Test,
                                  public C3CartpoleProblem {
 protected:
  ContactForceGeneratorTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {}

  void SetUp() override {
    generator_ = std::make_unique<ContactForceGenerator>();
    context_ = generator_->CreateDefaultContext();
    output_ = generator_->AllocateOutput();

    // Create mock C3 solution
    solution_ = std::make_unique<C3Output::C3Solution>(
        pSystem->num_states(), pSystem->num_lambdas(), pSystem->num_inputs(),
        pSystem->N());
    solution_->time_vector_ = VectorXf::LinSpaced(pSystem->N(), 0.0, 1.0);
    solution_->x_sol_ = MatrixXf::Random(pSystem->num_states(), pSystem->N());
    solution_->lambda_sol_ =
        MatrixXf::Random(pSystem->num_lambdas(), pSystem->N());
    solution_->u_sol_ = MatrixXf::Random(pSystem->num_inputs(), pSystem->N());

    // Create mock contact descriptions
    contact_descriptions_.resize(2);
    contact_descriptions_[0].witness_point_A = Eigen::Vector3d(1.0, 0.0, 0.0);
    contact_descriptions_[0].witness_point_B = Eigen::Vector3d(0.5, 0.0, 0.0);
    contact_descriptions_[0].force_basis = Eigen::Vector3d(1.0, 0.0, 0.0);
    contact_descriptions_[0].is_slack = false;

    contact_descriptions_[1].witness_point_A = Eigen::Vector3d(-1.0, 0.0, 0.0);
    contact_descriptions_[1].witness_point_B = Eigen::Vector3d(-0.5, 0.0, 0.0);
    contact_descriptions_[1].force_basis = Eigen::Vector3d(-1.0, 0.0, 0.0);
    contact_descriptions_[1].is_slack = true;  // This one should be ignored

    // Set up input ports
    generator_->get_input_port_c3_solution().FixValue(context_.get(),
                                                      *solution_);
    generator_->get_input_port_lcs_contact_descriptions().FixValue(
        context_.get(), contact_descriptions_);
  }

  std::unique_ptr<ContactForceGenerator> generator_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
  std::unique_ptr<C3Output::C3Solution> solution_;
  std::vector<LCSContactDescription> contact_descriptions_;
};

TEST_F(ContactForceGeneratorTest, InputOutputPortsExist) {
  // Check that all expected input and output ports exist
  EXPECT_NO_THROW(generator_->get_input_port_c3_solution());
  EXPECT_NO_THROW(generator_->get_input_port_lcs_contact_descriptions());
  EXPECT_NO_THROW(generator_->get_output_port_contact_force());
}

TEST_F(ContactForceGeneratorTest, GeneratesContactForces) {
  // Should not throw when generating contact forces
  EXPECT_NO_THROW(generator_->CalcOutput(*context_, output_.get()));

  // Get the output message
  const auto& forces_msg =
      output_->get_data(0)->get_value<c3::lcmt_contact_forces>();

  // Should have 1 force (one contact is slack and ignored)
  EXPECT_EQ(forces_msg.num_forces, 1);
  EXPECT_EQ(forces_msg.forces.size(), 1);

  // Check contact point coordinates
  const auto& force = forces_msg.forces[0];
  EXPECT_DOUBLE_EQ(force.contact_point[0], 0.5);
  EXPECT_DOUBLE_EQ(force.contact_point[1], 0.0);
  EXPECT_DOUBLE_EQ(force.contact_point[2], 0.0);

  // Check force calculation (force_basis * lambda)
  double expected_force = 1.0 * solution_->lambda_sol_(0, 0);
  EXPECT_DOUBLE_EQ(force.contact_force[0], expected_force);
  EXPECT_DOUBLE_EQ(force.contact_force[1], 0.0);
  EXPECT_DOUBLE_EQ(force.contact_force[2], 0.0);

  // Check timestamp
  EXPECT_DOUBLE_EQ(forces_msg.utime, context_->get_time() * 1e6);
}

TEST_F(ContactForceGeneratorTest, IgnoresSlackContacts) {
  // Mark all contacts as slack
  for (auto& contact : contact_descriptions_) {
    contact.is_slack = true;
  }
  generator_->get_input_port_lcs_contact_descriptions().FixValue(
      context_.get(), contact_descriptions_);

  generator_->CalcOutput(*context_, output_.get());
  const auto& forces_msg =
      output_->get_data(0)->get_value<c3::lcmt_contact_forces>();

  // Should have no forces when all contacts are slack
  EXPECT_EQ(forces_msg.num_forces, 0);
  EXPECT_TRUE(forces_msg.forces.empty());
}

// Test fixture for C3TrajectoryGenerator
class C3TrajectoryGeneratorTest : public ::testing::Test,
                                  public C3CartpoleProblem {
 protected:
  C3TrajectoryGeneratorTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {}

  void SetUp() override {
    // Set up trajectory generator config
    config_.trajectories.resize(3);

    // State trajectory
    config_.trajectories[0].trajectory_name = "states";
    config_.trajectories[0].variable_type = "state";
    config_.trajectories[0].indices = {{.start = 0, .end = 1}};

    // Input trajectory
    config_.trajectories[1].trajectory_name = "inputs";
    config_.trajectories[1].variable_type = "input";
    config_.trajectories[1].indices = {{.start = 0, .end = 0}};

    // Force trajectory
    config_.trajectories[2].trajectory_name = "forces";
    config_.trajectories[2].variable_type = "force";
    // Empty indices should use default (all)

    generator_ = std::make_unique<C3TrajectoryGenerator>(config_);
    context_ = generator_->CreateDefaultContext();
    output_ = generator_->AllocateOutput();

    // Create mock C3 solution
    solution_ = std::make_unique<C3Output::C3Solution>(
        pSystem->num_states(), pSystem->num_lambdas(), pSystem->num_inputs(),
        pSystem->N());
    solution_->time_vector_ = VectorXf::LinSpaced(pSystem->N(), 0.0, 1.0);
    solution_->x_sol_ = MatrixXf::Random(pSystem->num_states(), pSystem->N());
    solution_->lambda_sol_ =
        MatrixXf::Random(pSystem->num_lambdas(), pSystem->N());
    solution_->u_sol_ = MatrixXf::Random(pSystem->num_inputs(), pSystem->N());

    // Set up input port
    generator_->get_input_port_c3_solution().FixValue(context_.get(),
                                                      *solution_);
  }

  C3TrajectoryGeneratorConfig config_;
  std::unique_ptr<C3TrajectoryGenerator> generator_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
  std::unique_ptr<C3Output::C3Solution> solution_;
};

TEST_F(C3TrajectoryGeneratorTest, InputOutputPortsExist) {
  // Check that all expected input and output ports exist
  EXPECT_NO_THROW(generator_->get_input_port_c3_solution());
  EXPECT_NO_THROW(generator_->get_output_port_lcmt_c3_trajectory());
}

TEST_F(C3TrajectoryGeneratorTest, GeneratesTrajectories) {
  // Should not throw when generating trajectories
  EXPECT_NO_THROW(generator_->CalcOutput(*context_, output_.get()));

  // Get the output message
  const auto& traj_msg =
      output_->get_data(0)->get_value<c3::lcmt_c3_trajectory>();

  // Should have 3 trajectories as configured
  EXPECT_EQ(traj_msg.num_trajectories, 3);
  EXPECT_EQ(traj_msg.trajectories.size(), 3);

  // Check state trajectory
  const auto& state_traj = traj_msg.trajectories[0];
  EXPECT_EQ(state_traj.trajectory_name, "states");
  EXPECT_EQ(state_traj.num_timesteps, pSystem->N());
  EXPECT_EQ(state_traj.vector_dim, 2);  // indices 0-1
  EXPECT_EQ(state_traj.timestamps.size(), pSystem->N());
  EXPECT_EQ(state_traj.values.size(), 2);

  // Check input trajectory
  const auto& input_traj = traj_msg.trajectories[1];
  EXPECT_EQ(input_traj.trajectory_name, "inputs");
  EXPECT_EQ(input_traj.num_timesteps, pSystem->N());
  EXPECT_EQ(input_traj.vector_dim, 1);  // indices 0-0
  EXPECT_EQ(input_traj.values.size(), 1);

  // Check force trajectory (should use all lambda indices)
  const auto& force_traj = traj_msg.trajectories[2];
  EXPECT_EQ(force_traj.trajectory_name, "forces");
  EXPECT_EQ(force_traj.num_timesteps, pSystem->N());
  EXPECT_EQ(force_traj.vector_dim, pSystem->num_lambdas());
  EXPECT_EQ(force_traj.values.size(), pSystem->num_lambdas());

  // Check timestamp
  EXPECT_DOUBLE_EQ(traj_msg.utime, context_->get_time() * 1e6);
}

TEST_F(C3TrajectoryGeneratorTest, ThrowsOnInvalidVariableType) {
  // Create config with invalid variable type
  C3TrajectoryGeneratorConfig bad_config;
  bad_config.trajectories.resize(1);
  bad_config.trajectories[0].trajectory_name = "invalid";
  bad_config.trajectories[0].variable_type = "invalid_type";

  auto bad_generator = std::make_unique<C3TrajectoryGenerator>(bad_config);
  auto bad_context = bad_generator->CreateDefaultContext();
  auto bad_output = bad_generator->AllocateOutput();

  bad_generator->get_input_port_c3_solution().FixValue(bad_context.get(),
                                                       *solution_);

  // Should throw when trying to generate trajectories with invalid type
  EXPECT_THROW(bad_generator->CalcOutput(*bad_context, bad_output.get()),
               std::runtime_error);
}

TEST_F(C3TrajectoryGeneratorTest, ThrowsOnOutOfRangeIndices) {
  // Create config with out-of-range indices
  C3TrajectoryGeneratorConfig bad_config;
  bad_config.trajectories.resize(1);
  bad_config.trajectories[0].trajectory_name = "out_of_range";
  bad_config.trajectories[0].variable_type = "state";
  bad_config.trajectories[0].indices = {{.start = 0, .end = 999}};  // Too high

  auto bad_generator = std::make_unique<C3TrajectoryGenerator>(bad_config);
  auto bad_context = bad_generator->CreateDefaultContext();
  auto bad_output = bad_generator->AllocateOutput();

  bad_generator->get_input_port_c3_solution().FixValue(bad_context.get(),
                                                       *solution_);

  // Should throw when indices are out of range
  EXPECT_THROW(bad_generator->CalcOutput(*bad_context, bad_output.get()),
               std::out_of_range);
}

// Test fixture for C3OutputGenerator
class C3OutputGeneratorTest : public ::testing::Test, public C3CartpoleProblem {
 protected:
  C3OutputGeneratorTest()
      : C3CartpoleProblem(0.411, 0.978, 0.6, 0.4267, 0.35, -0.35, 100, 9.81) {}

  void SetUp() override {
    generator_ = std::make_unique<C3OutputGenerator>();
    context_ = generator_->CreateDefaultContext();
    output_ = generator_->AllocateOutput();

    // Create mock C3 solution
    solution_ = std::make_unique<C3Output::C3Solution>(
        pSystem->num_states(), pSystem->num_lambdas(), pSystem->num_inputs(),
        pSystem->N());
    solution_->time_vector_ = VectorXf::LinSpaced(pSystem->N(), 0.0, 1.0);
    solution_->x_sol_ = MatrixXf::Random(pSystem->num_states(), pSystem->N());
    solution_->lambda_sol_ =
        MatrixXf::Random(pSystem->num_lambdas(), pSystem->N());
    solution_->u_sol_ = MatrixXf::Random(pSystem->num_inputs(), pSystem->N());

    // Create mock C3 intermediates
    int total_vars =
        pSystem->num_states() + pSystem->num_lambdas() + pSystem->num_inputs();
    intermediates_ = std::make_unique<C3Output::C3Intermediates>(
        pSystem->num_states(), pSystem->num_lambdas(), pSystem->num_inputs(),
        pSystem->N());
    intermediates_->time_vector_ = VectorXf::LinSpaced(pSystem->N(), 0.0, 1.0);
    intermediates_->z_ = MatrixXf::Random(total_vars, pSystem->N());
    intermediates_->delta_ = MatrixXf::Random(total_vars, pSystem->N());
    intermediates_->w_ = MatrixXf::Random(total_vars, pSystem->N());

    // Set up input ports
    generator_->get_input_port_c3_solution().FixValue(context_.get(),
                                                      *solution_);
    generator_->get_input_port_c3_intermediates().FixValue(context_.get(),
                                                           *intermediates_);
  }

  std::unique_ptr<C3OutputGenerator> generator_;
  std::unique_ptr<Context<double>> context_;
  std::unique_ptr<SystemOutput<double>> output_;
  std::unique_ptr<C3Output::C3Solution> solution_;
  std::unique_ptr<C3Output::C3Intermediates> intermediates_;
};

TEST_F(C3OutputGeneratorTest, InputOutputPortsExist) {
  // Check that all expected input and output ports exist
  EXPECT_NO_THROW(generator_->get_input_port_c3_solution());
  EXPECT_NO_THROW(generator_->get_input_port_c3_intermediates());
  EXPECT_NO_THROW(generator_->get_output_port_c3_output());
}

TEST_F(C3OutputGeneratorTest, GeneratesOutput) {
  // Should not throw when generating output
  EXPECT_NO_THROW(generator_->CalcOutput(*context_, output_.get()));

  // Get the output message
  const auto& output_msg = output_->get_data(0)->get_value<c3::lcmt_output>();

  // Check basic properties
  EXPECT_DOUBLE_EQ(output_msg.utime, context_->get_time() * 1e6);

  // Check that solution data is present
  EXPECT_EQ(output_msg.solution.time_vec.size(), pSystem->N());
  EXPECT_EQ(output_msg.solution.x_sol.size(), pSystem->num_states());
  EXPECT_EQ(output_msg.solution.x_sol[0].size(), pSystem->N());
  EXPECT_EQ(output_msg.solution.lambda_sol.size(), pSystem->num_lambdas());
  EXPECT_EQ(output_msg.solution.lambda_sol[0].size(), pSystem->N());
  EXPECT_EQ(output_msg.solution.u_sol.size(), pSystem->num_inputs());
  EXPECT_EQ(output_msg.solution.u_sol[0].size(), pSystem->N());

  // Check that intermediates data is present
  int total_vars =
      pSystem->num_states() + pSystem->num_lambdas() + pSystem->num_inputs();
  EXPECT_EQ(output_msg.intermediates.z_sol.size(), total_vars);
  EXPECT_EQ(output_msg.intermediates.z_sol[0].size(), pSystem->N());
  EXPECT_EQ(output_msg.intermediates.delta_sol.size(), total_vars);
  EXPECT_EQ(output_msg.intermediates.delta_sol[0].size(), pSystem->N());
  EXPECT_EQ(output_msg.intermediates.w_sol.size(), total_vars);
  EXPECT_EQ(output_msg.intermediates.w_sol[0].size(), pSystem->N());
  EXPECT_EQ(output_msg.intermediates.time_vec.size(), pSystem->N());
}

TEST_F(C3OutputGeneratorTest, VerifyDataConsistency) {
  generator_->CalcOutput(*context_, output_.get());
  const auto& output_msg = output_->get_data(0)->get_value<c3::lcmt_output>();

  // Check that time vectors match
  for (int i = 0; i < pSystem->N(); ++i) {
    EXPECT_FLOAT_EQ(output_msg.solution.time_vec[i],
                    solution_->time_vector_(i));
  }

  // Check that solution data matches (spot check first few elements)
  for (int i = 0; i < std::min(3, (int)output_msg.solution.x_sol.size()); ++i) {
    int row = i % pSystem->num_states();
    int col = i / pSystem->num_states();
    EXPECT_FLOAT_EQ(output_msg.solution.x_sol[row][col],
                    solution_->x_sol_(row, col));
  }
}

}  // namespace test
}  // namespace systems
}  // namespace c3