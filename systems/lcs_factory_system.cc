#include "lcs_factory_system.h"

#include <utility>

#include "core/lcs.h"
#include "multibody/lcs_factory.h"
#include "multibody/multibody_utils.h"
#include "systems/framework/timestamped_vector.h"

using c3::LCS;
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
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>
        contact_geoms,
    LCSOptions options)
    : plant_(plant),
      context_(context),
      plant_ad_(plant_ad),
      context_ad_(context_ad),
      contact_pairs_(contact_geoms),
      options_(std::move(options)),
      N_(options_.N),
      dt_(options_.dt) {
  this->set_name("lcs_factory_system");

  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_x_ = n_q_ + n_v_;
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(options_);
  n_u_ = plant_.num_actuators();

  lcs_state_input_port_ =
      this->DeclareVectorInputPort("x_lcs", TimestampedVector<double>(n_x_))
          .get_index();

  lcs_inputs_input_port_ =
      this->DeclareVectorInputPort("u_lcs", BasicVector<double>(n_u_))
          .get_index();

  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, N_, dt_);
  lcs_port_ = this->DeclareAbstractOutputPort("lcs", lcs_placeholder,
                                              &LCSFactorySystem::OutputLCS)
                  .get_index();

  lcs_contact_jacobian_port_ =
      this->DeclareAbstractOutputPort(
              "J_lcs, p_lcs",
              std::pair(Eigen::MatrixXd(n_x_, n_lambda_),
                        std::vector<Eigen::VectorXd>()),
              &LCSFactorySystem::OutputLCSContactJacobian)
          .get_index();
}

void LCSFactorySystem::OutputLCS(const drake::systems::Context<double>& context,
                                 LCS* output_lcs) const {
  const auto lcs_x = (TimestampedVector<double>*)this->EvalVectorInput(
      context, lcs_state_input_port_);
  const auto lcs_u = (BasicVector<double>*)this->EvalVectorInput(
      context, lcs_inputs_input_port_);
  DRAKE_DEMAND(lcs_x->get_data().size() == n_x_);
  DRAKE_DEMAND(lcs_u->get_value().size() == n_u_);

  VectorXd q_v_u = VectorXd::Zero(n_x_ + n_u_);
  q_v_u << lcs_x->get_data(), lcs_u->get_value();
  drake::AutoDiffVecXd q_v_u_ad = drake::math::InitializeAutoDiff(q_v_u);

  multibody::SetPositionsAndVelocitiesIfNew<double>(plant_, q_v_u.head(4),
                                                    &context_);
  multibody::SetInputsIfNew<double>(plant_, q_v_u.tail(n_u_), &context_);
  multibody::SetPositionsAndVelocitiesIfNew<drake::AutoDiffXd>(
      plant_ad_, q_v_u_ad.head(4), &context_ad_);
  multibody::SetInputsIfNew<drake::AutoDiffXd>(plant_ad_, q_v_u_ad.tail(n_u_),
                                               &context_ad_);

  *output_lcs = LCSFactory::LinearizePlantToLCS(
      plant_, context_, plant_ad_, context_ad_, contact_pairs_, options_);
}

void LCSFactorySystem::OutputLCSContactJacobian(
    const drake::systems::Context<double>& context,
    std::pair<Eigen::MatrixXd, std::vector<Eigen::VectorXd>>* output) const {
  const TimestampedVector<double>* lcs_x =
      (TimestampedVector<double>*)this->EvalVectorInput(context,
                                                        lcs_state_input_port_);

  // u is irrelevant in pure geometric/kinematic calculation
  VectorXd state = lcs_x->get_data();
  multibody::SetPositionsAndVelocitiesIfNew<double>(plant_, state, &context_);

  std::vector<Eigen::VectorXd> contact_points;
  *output = LCSFactory::ComputeContactJacobian(plant_, context_, contact_pairs_,
                                               options_);
}

}  // namespace systems
}  // namespace c3
