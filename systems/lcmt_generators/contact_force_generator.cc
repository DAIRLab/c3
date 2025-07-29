#include "systems/lcmt_generators/contact_force_generator.h"

#include <algorithm>

#include "multibody/lcs_factory.h"
#include "systems/framework/c3_output.h"

// Drake and Eigen namespace usages for convenience.
using drake::systems::Context;
using drake::systems::DiagramBuilder;
using drake::systems::lcm::LcmPublisherSystem;

using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::VectorXd;

namespace c3 {

using multibody::LCSContactDescription;

namespace systems {
namespace lcmt_generators {

// Publishes contact force information to LCM for visualization and logging.
ContactForceGenerator::ContactForceGenerator() {
  // Declare input port for the C3 solution.
  c3_solution_port_ = this->DeclareAbstractInputPort(
                              "solution", drake::Value<C3Output::C3Solution>{})
                          .get_index();
  // Declare input port for contact Jacobian and contact points.
  lcs_contact_info_port_ =
      this->DeclareAbstractInputPort(
              "contact_descriptions",
              drake::Value<std::vector<LCSContactDescription>>())
          .get_index();

  this->set_name("c3_contact_force_generator");
  // Declare output port for publishing contact forces.
  contact_force_output_port_ =
      this->DeclareAbstractOutputPort("lcmt_force", lcmt_contact_forces(),
                                      &ContactForceGenerator::DoCalc)
          .get_index();
}

// Calculates and outputs the contact forces based on the current context.
void ContactForceGenerator::DoCalc(const Context<double>& context,
                                   lcmt_contact_forces* output) const {
  // Get Solution from C3
  const auto& solution =
      this->EvalInputValue<C3Output::C3Solution>(context, c3_solution_port_);
  // Get Contact infromation form LCS Factory
  const auto& contact_info =
      this->EvalInputValue<std::vector<LCSContactDescription>>(
          context, lcs_contact_info_port_);

  output->forces.clear();
  output->num_forces = 0;
  // Iterate over all contact points and compute forces.
  for (size_t i = 0; i < contact_info->size(); ++i) {
    auto force = lcmt_contact_force();

    if (contact_info->at(i).is_slack)
      // If the contact is slack, ignore it.
      continue;

    // Set contact point position.
    force.contact_point[0] = contact_info->at(i).witness_point_B[0];
    force.contact_point[1] = contact_info->at(i).witness_point_B[1];
    force.contact_point[2] = contact_info->at(i).witness_point_B[2];

    // If the contact is not slack, compute the force.
    force.contact_force[0] =
        contact_info->at(i).force_basis[0] * solution->lambda_sol_(i, 0);
    force.contact_force[1] =
        contact_info->at(i).force_basis[1] * solution->lambda_sol_(i, 0);
    force.contact_force[2] =
        contact_info->at(i).force_basis[2] * solution->lambda_sol_(i, 0);

    output->forces.push_back(force);
  }
  // Set output timestamp in microseconds.
  output->num_forces = output->forces.size();
  output->utime = context.get_time() * 1e6;
}

// Adds this publisher and an LCM publisher system to the diagram builder.
LcmPublisherSystem* ContactForceGenerator::AddLcmPublisherToBuilder(
    DiagramBuilder<double>& builder,
    const drake::systems::OutputPort<double>& solution_port,
    const drake::systems::OutputPort<double>& lcs_contact_descriptions_port,
    const std::string& channel, drake::lcm::DrakeLcmInterface* lcm,
    const drake::systems::TriggerTypeSet& publish_triggers,
    double publish_period, double publish_offset) {
  // Add and connect the ContactForceGenerator system.
  auto force_publisher = builder.AddSystem<ContactForceGenerator>();
  builder.Connect(solution_port, force_publisher->get_input_port_c3_solution());
  builder.Connect(lcs_contact_descriptions_port,
                  force_publisher->get_input_port_lcs_contact_descriptions());

  // Add and connect the LCM publisher system.
  auto lcm_force_publisher =
      builder.AddSystem(LcmPublisherSystem::Make<c3::lcmt_contact_forces>(
          channel, lcm, publish_triggers, publish_period, publish_offset));
  builder.Connect(force_publisher->get_output_port_contact_force(),
                  lcm_force_publisher->get_input_port());
  return lcm_force_publisher;
}

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3