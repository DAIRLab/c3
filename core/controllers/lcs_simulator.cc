#include "lcs_simulator.h"

#include <Eigen/Dense>
#include <set>

using Eigen::VectorXd;

namespace c3 {
namespace systems {

LCSSimulator::LCSSimulator(int n_x, int n_u, int n_lambda, int N, double dt) {

  init(n_x, n_u, n_lambda, N, dt);
}

LCSSimulator::LCSSimulator(LCS *lcs) {
  DRAKE_DEMAND(lcs != nullptr);
  init(lcs->num_states(), lcs->num_inputs(), lcs->num_lambdas(), lcs->N(),
       lcs->dt());
}

void LCSSimulator::init(int n_x, int n_u, int n_lambda, int N, double dt) {

  LCS placeholder_lcs = LCS::CreatePlaceholderLCS(n_x, n_u, n_lambda, N, dt);

  lcs_input_port_ =
      this->DeclareAbstractInputPort("lcs", drake::Value<LCS>(placeholder_lcs))
          .get_index();
  state_input_port_ = this->DeclareVectorInputPort(
                              "x", drake::systems::BasicVector<double>(n_x))
                          .get_index();
  action_input_port_ = this->DeclareVectorInputPort(
                               "u", drake::systems::BasicVector<double>(n_u))
                           .get_index();
                           
  next_state_output_port_ =
      this->DeclareVectorOutputPort("x_next", n_x,
                                    &LCSSimulator::SimulateOneStep)
          .get_index();
}

void LCSSimulator::SimulateOneStep(
    const drake::systems::Context<double> &context,
    drake::systems::BasicVector<double> *next_state) const {

  LCS lcs = this->EvalAbstractInput(context, lcs_input_port_)->get_value<LCS>();
  auto state = this->EvalVectorInput(context, state_input_port_)->value();
  auto action = this->EvalVectorInput(context, action_input_port_)->value();
  VectorXd x = Eigen::Map<VectorXd>(state.data(), state.size());
  VectorXd u = Eigen::Map<VectorXd>(action.data(), action.size());
  next_state->set_value(lcs.Simulate(x, u));
}

} // namespace systems

} // namespace c3