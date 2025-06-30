#include "force_publisher.h"

#include "systems/framework/c3_output.h"

// Drake and Eigen namespace usages for convenience.
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::lcm::LcmPublisherSystem;

using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::VectorXd;

namespace c3 {
namespace systems {
namespace publishers {

// Publishes contact force information to LCM for visualization and logging.
ContactForcePublisher::ContactForcePublisher() {
  // Declare input port for the C3 solution.
  c3_solution_port_ = this->DeclareAbstractInputPort(
                              "solution", drake::Value<C3Output::C3Solution>{})
                          .get_index();
  // Declare input port for contact Jacobian and contact points.
  lcs_contact_info_port_ =
      this->DeclareAbstractInputPort(
              "J_lcs, p_lcs",
              drake::Value<
                  std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>>())
          .get_index();

  this->set_name("c3_contact_force_publisher");
  // Declare output port for publishing contact forces.
  contact_force_output_port_ =
      this->DeclareAbstractOutputPort("lcmt_force", lcmt_forces(),
                                      &ContactForcePublisher::DoCalc)
          .get_index();
}

// Calculates and outputs the contact forces based on the current context.
void ContactForcePublisher::DoCalc(const Context<double>& context,
                                   lcmt_forces* output) const {
  // Get Solution from C3
  const auto& solution =
      this->EvalInputValue<C3Output::C3Solution>(context, c3_solution_port_);
  // Get Contact infromation form LCS Factory
  const auto& contact_info = this->EvalInputValue<
      std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>>(
      context, lcs_contact_info_port_);
  // Contact Jacobian
  MatrixXd J_c = contact_info->first;
  int contact_force_start = solution->lambda_sol_.rows() - J_c.rows();
  bool using_stewart_and_trinkle_model = contact_force_start > 0;

  auto contact_points = contact_info->second;
  int forces_per_contact = contact_info->first.rows() / contact_points.size();

  output->num_forces = forces_per_contact * contact_points.size();
  output->forces.resize(output->num_forces);

  int contact_var_start;
  int force_index;
  // Iterate over all contact points and compute forces.
  for (int contact_index = 0; contact_index < (int)contact_points.size();
       ++contact_index) {
    contact_var_start =
        contact_force_start + forces_per_contact * contact_index;
    force_index = forces_per_contact * contact_index;
    for (int i = 0; i < forces_per_contact; ++i) {
      int contact_jacobian_row = force_index + i;  // index for anitescu model
      int contact_var_index = contact_var_start + i;
      if (using_stewart_and_trinkle_model) {  // index for stweart and trinkle
                                              // model
        if (i == 0) {
          contact_jacobian_row = contact_index;
          contact_var_index = contact_force_start + contact_index;
        } else {
          contact_jacobian_row = contact_points.size() +
                                 (forces_per_contact - 1) * contact_index + i -
                                 1;
          contact_var_index = contact_force_start + contact_points.size() +
                              (forces_per_contact - 1) * contact_index + i - 1;
        }
      }
      auto force = lcmt_contact_force();
      // Set contact point position.
      force.contact_point[0] = contact_points.at(contact_index)[0];
      force.contact_point[1] = contact_points.at(contact_index)[1];
      force.contact_point[2] = contact_points.at(contact_index)[2];
      // TODO(yangwill): find a cleaner way to figure out the equivalent forces
      // VISUALIZING FORCES FOR THE FIRST KNOT POINT
      // 6, 7, 8 are the indices for the x,y,z components of the tray
      // expressed in the world frame
      // Compute force vector in world frame.
      std::cout << J_c << std::endl;
      exit(0);
      force.contact_force[0] = solution->lambda_sol_(contact_var_index, 0) *
                               J_c.row(contact_jacobian_row)(6);
      force.contact_force[1] = solution->lambda_sol_(contact_var_index, 0) *
                               J_c.row(contact_jacobian_row)(7);
      force.contact_force[2] = solution->lambda_sol_(contact_var_index, 0) *
                               J_c.row(contact_jacobian_row)(8);
      output->forces[force_index + i] = force;
    }
  }
  // Set output timestamp in microseconds.
  output->utime = context.get_time() * 1e6;
}

// Adds this publisher and an LCM publisher system to the diagram builder.
LcmPublisherSystem* ContactForcePublisher::AddLcmPublisherToBuilder(
    DiagramBuilder<double>& builder,
    const drake::systems::OutputPort<double>& solution_port,
    const drake::systems::OutputPort<double>& lcs_contact_info_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  // Add and connect the ContactForcePublisher system.
  auto force_publisher = builder.AddSystem<ContactForcePublisher>();
  builder.Connect(solution_port, force_publisher->get_input_port_c3_solution());
  builder.Connect(lcs_contact_info_port,
                  force_publisher->get_input_port_lcs_contact_info());

  // Add and connect the LCM publisher system.
  auto lcm_force_publisher =
      builder.AddSystem(LcmPublisherSystem::Make<c3::lcmt_forces>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(force_publisher->get_output_port_contact_force(),
                  lcm_force_publisher->get_input_port());
  return lcm_force_publisher;
}

}  // namespace publishers
}  // namespace systems
}  // namespace c3