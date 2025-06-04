#include "lcs_simulator.h"

#include <set>

#include <Eigen/Dense>

using Eigen::VectorXd;

namespace c3 {
namespace systems {

// Constructor with parameter initialization
LCSSimulator::LCSSimulator(int n_x, int n_u, int n_lambda, int N, double dt) {
  init(n_x, n_u, n_lambda, N, dt);
}

// Constructor using an existing LCS object
LCSSimulator::LCSSimulator(LCS lcs) {
  init(lcs.num_states(), lcs.num_inputs(), lcs.num_lambdas(), lcs.N(),
       lcs.dt());
}

// Initialization function for setting up ports and placeholder LCS
void LCSSimulator::init(int n_x, int n_u, int n_lambda, int N, double dt) {
  // Create a placeholder LCS object
  LCS placeholder_lcs = LCS::CreatePlaceholderLCS(n_x, n_u, n_lambda, N, dt);

  // Declare input and output ports
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

// Simulates one step of the LCS system
void LCSSimulator::SimulateOneStep(
    const drake::systems::Context<double>& context,
    drake::systems::BasicVector<double>* next_state) const {
  // Retrieve the LCS object from the input port
  LCS lcs = this->EvalAbstractInput(context, lcs_input_port_)->get_value<LCS>();

  // Retrieve the current state and action from the input ports
  auto state = this->EvalVectorInput(context, state_input_port_)->value();
  auto action = this->EvalVectorInput(context, action_input_port_)->value();

  // Map state and action to Eigen vectors
  VectorXd x = Eigen::Map<VectorXd>(state.data(), state.size());
  VectorXd u = Eigen::Map<VectorXd>(action.data(), action.size());

  // Compute the next state using the LCS simulation
  next_state->set_value(lcs.Simulate(x, u));
}

}  // namespace systems
}  // namespace c3
