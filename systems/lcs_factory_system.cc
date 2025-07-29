#include "lcs_factory_system.h"

#include <utility>

#include "core/lcs.h"
#include "multibody/lcs_factory.h"
#include "multibody/multibody_utils.h"
#include "systems/framework/timestamped_vector.h"

using c3::LCS;
using c3::multibody::LCSContactDescription;
using c3::multibody::LCSFactory;
using c3::systems::TimestampedVector;
using drake::multibody::ModelInstanceIndex;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;

namespace c3 {

namespace systems {

LCSFactorySystem::LCSFactorySystem(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    LCSFactoryOptions options) {
  InitializeSystem(plant, options);
  lcs_factory_ = std::make_unique<multibody::LCSFactory>(
      plant, context, plant_ad, context_ad, options);
}

LCSFactorySystem::LCSFactorySystem(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>
        contact_geoms,
    LCSFactoryOptions options) {
  InitializeSystem(plant, options);
  lcs_factory_ = std::make_unique<multibody::LCSFactory>(
      plant, context, plant_ad, context_ad, contact_geoms, options);
}

void LCSFactorySystem::InitializeSystem(
    const drake::multibody::MultibodyPlant<double>& plant,
    LCSFactoryOptions& options) {
  this->set_name("lcs_factory_system");

  n_x_ = plant.num_positions() + plant.num_velocities();
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(options);
  n_u_ = plant.num_actuators();

  lcs_state_input_port_ =
      this->DeclareVectorInputPort("x_lcs", TimestampedVector<double>(n_x_))
          .get_index();

  lcs_inputs_input_port_ =
      this->DeclareVectorInputPort("u_lcs", BasicVector<double>(n_u_))
          .get_index();

  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, options.N, options.dt);
  lcs_port_ = this->DeclareAbstractOutputPort("lcs", lcs_placeholder,
                                              &LCSFactorySystem::OutputLCS)
                  .get_index();

  lcs_contact_descriptions_output_port_ =
      this->DeclareAbstractOutputPort(
              "contact_descriptions", std::vector<LCSContactDescription>(),
              &LCSFactorySystem::OutputLCSContactDescriptions)
          .get_index();
}

void LCSFactorySystem::OutputLCS(const drake::systems::Context<double>& context,
                                 LCS* output_lcs) const {
  const auto lcs_x = static_cast<const TimestampedVector<double>*>(
      this->EvalVectorInput(context, lcs_state_input_port_));
  const auto lcs_u = static_cast<const BasicVector<double>*>(
      this->EvalVectorInput(context, lcs_inputs_input_port_));
  DRAKE_DEMAND(lcs_x->get_data().size() == n_x_);
  DRAKE_DEMAND(lcs_u->get_value().size() == n_u_);

  lcs_factory_->UpdateStateAndInput(lcs_x->get_data(), lcs_u->get_value());
  *output_lcs = lcs_factory_->GenerateLCS();
}

void LCSFactorySystem::OutputLCSContactDescriptions(
    const drake::systems::Context<double>& context,
    std::vector<LCSContactDescription>* output) const {
  const auto lcs_x = static_cast<const TimestampedVector<double>*>(
      this->EvalVectorInput(context, lcs_state_input_port_));
  const auto lcs_u = static_cast<const BasicVector<double>*>(
      this->EvalVectorInput(context, lcs_inputs_input_port_));

  DRAKE_DEMAND(lcs_x->get_data().size() == n_x_);
  DRAKE_DEMAND(lcs_u->get_value().size() == n_u_);
  lcs_factory_->UpdateStateAndInput(lcs_x->get_data(), lcs_u->get_value());
  *output = lcs_factory_->GetContactDescriptions();
}

}  // namespace systems
}  // namespace c3
