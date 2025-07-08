#include "systems/lcmt_generators/c3_trajectory_generator.h"

#include "systems/framework/c3_output.h"

namespace c3 {
namespace systems {
namespace lcmt_generators {

C3TrajectoryGenerator::C3TrajectoryGenerator(C3TrajectoryGeneratorConfig config)
    : config_(config) {
  // Declare input ports for solution and intermediates.
  c3_solution_input_port_ =
      this->DeclareAbstractInputPort("c3_solution",
                                     drake::Value<C3Output::C3Solution>())
          .get_index();

  // Declare output port for lcmt_trajectory message.
  lcmt_c3_trajectory_output_port_ =
      this->DeclareAbstractOutputPort(
              "lcmt_c3_trajectory", &C3TrajectoryGenerator::GenerateTrajectory)
          .get_index();

  // this->set_name("c3_trajectory_generator");
}

void C3TrajectoryGenerator::GenerateTrajectory(
    const drake::systems::Context<double>& context,
    lcmt_c3_trajectory* output) const {
  // Get input data from the input port
  const auto* solution = this->EvalInputValue<C3Output::C3Solution>(
      context, c3_solution_input_port_);

  Eigen::VectorXf time_vector = solution->GetTimeVector();
  Eigen::MatrixXf solution_values;
  output->num_trajectories = config_.trajectories.size();
  output->trajectories.clear();

  // Iterate over each trajectory description in config
  for (const auto& traj_desc : config_.trajectories) {
    lcmt_trajectory trajectory;
    trajectory.trajectory_name = traj_desc.trajectory_name;

    // Select the appropriate solution matrix based on variable type
    if (traj_desc.variable_type == "state") {
      solution_values = solution->GetStateSolution();
    } else if (traj_desc.variable_type == "input") {
      solution_values = solution->GetInputSolution();
    } else if (traj_desc.variable_type == "force") {
      solution_values = solution->GetForceSolution();
    } else {
      throw std::runtime_error(
          "Unknown variable type in C3 trajectory description.");
    }

    DRAKE_ASSERT(time_vector.size() == solution_values.cols());

    // Copy timestamps for all timesteps
    trajectory.num_timesteps = solution_values.cols();
    trajectory.timestamps.reserve(trajectory.num_timesteps);
    trajectory.timestamps.assign(time_vector.data(),
                                 time_vector.data() + trajectory.num_timesteps);

    trajectory.values.clear();
    // Extract each index range specified in the trajectory description
    auto indices = traj_desc.indices;
    if (indices.empty()) {
      TrajectoryDescription::index_range default_indices = {
          .start = 0, .end = (int)solution_values.rows() - 1};
      indices.push_back(default_indices);
    }

    for (const auto& i : indices) {
      int start = i.start;
      int end = i.end;
      if (start < 0 || end >= solution_values.rows()) {
        throw std::out_of_range("Index out of range in C3 solution.");
      }

      // Copy the relevant rows from the solution matrix
      for (int i = start; i <= end; ++i) {
        std::vector<float> row(trajectory.num_timesteps);
        Eigen::VectorXf solution_row = solution_values.row(i);
        memcpy(row.data(), solution_row.data(),
               sizeof(float) * trajectory.num_timesteps);
        trajectory.values.push_back(row);
      }
    }
    // Set the number of timesteps and vector dimension for the trajectory
    trajectory.vector_dim = (int32_t)trajectory.values.size();
    // Add the constructed trajectory to the output message
    output->trajectories.push_back(trajectory);
  }

  // Set the timestamp for the output message
  output->utime = context.get_time() * 1e6;
}

drake::systems::lcm::LcmPublisherSystem*
C3TrajectoryGenerator::AddLcmPublisherToBuilder(
    drake::systems::DiagramBuilder<double>& builder,
    const C3TrajectoryGeneratorConfig config,
    const drake::systems::OutputPort<double>& c3_solution_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  auto traj_gen = builder.AddSystem<C3TrajectoryGenerator>(config);
  builder.Connect(c3_solution_port, traj_gen->get_input_port_c3_solution());

  auto lcm_pub = builder.AddSystem(
      drake::systems::lcm::LcmPublisherSystem::Make<lcmt_c3_trajectory>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(traj_gen->get_output_port_lcmt_c3_trajectory(),
                  lcm_pub->get_input_port());
  return lcm_pub;
}

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3