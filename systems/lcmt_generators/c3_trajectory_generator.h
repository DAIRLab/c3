#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/systems/lcm/lcm_interface_system.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

#include "c3/lcmt_c3_trajectory.hpp"
#include "c3/lcmt_trajectory.hpp"
#include "systems/lcmt_generators/c3_trajectory_generator_config.h"

namespace c3 {
namespace systems {
namespace lcmt_generators {

/**
 * @class C3TrajectoryGenerator
 * @brief Drake LeafSystem for generating lcmt_trajectory messages from input
 * data.
 *
 * The C3TrajectoryGenerator system takes trajectory solution and intermediate
 * data as input, and produces an lcmt_trajectory message as output. It is
 * configurable via the TrajectoryGeneratorOptions struct.
 */
class C3TrajectoryGenerator : public drake::systems::LeafSystem<double> {
 public:
  /**
   * @brief Constructor. Declares input and output ports.
   * @param config Configuration options for the trajectory generator.
   */
  C3TrajectoryGenerator(C3TrajectoryGeneratorConfig config);

  /**
   * @brief Returns the input port for the trajectory solution data.
   * @return Reference to the input port for c3 solution data.
   */
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_input_port_);
  }

  /**
   * @brief Returns the output port for the generated lcmt_trajectory message.
   * @return Reference to the output port for lcmt_trajectory.
   */
  const drake::systems::OutputPort<double>& get_output_port_lcmt_c3_trajectory()
      const {
    return this->get_output_port(lcmt_c3_trajectory_output_port_);
  }

  static drake::systems::lcm::LcmPublisherSystem* AddLcmPublisherToBuilder(
      drake::systems::DiagramBuilder<double>& builder,
      const C3TrajectoryGeneratorConfig config,
      const drake::systems::OutputPort<double>& c3_solution_port,
      const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
      const drake::systems::TriggerTypeSet& publish_triggers,
      double publish_period = 0.0, double publish_offset = 0.0);

 private:
  // Input port index for the trajectory solution data.
  drake::systems::InputPortIndex c3_solution_input_port_;

  // Output port index for the generated lcmt_trajectory message.
  drake::systems::OutputPortIndex lcmt_c3_trajectory_output_port_;

  // Options for configuring the trajectory generator.
  C3TrajectoryGeneratorConfig config_;

  /**
   * @brief Generates the lcmt_trajectory message from input data.
   * @param context The system context containing input values.
   * @param output Pointer to the lcmt_trajectory message to populate.
   */
  void GenerateTrajectory(const drake::systems::Context<double>& context,
                          lcmt_c3_trajectory* output) const;
};

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3