#pragma once

#include "core/lcs.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/leaf_system.h"

namespace c3 {
namespace systems {
class LCSSimulator : public drake::systems::LeafSystem<double> {
public:
  LCSSimulator(int n_x, int n_u, int n_lambda, int N, double dt);
  explicit LCSSimulator(LCS *lcs);

  const drake::systems::InputPort<double> &get_input_port_lcs() const {
    return this->get_input_port(lcs_input_port_);
  }

  const drake::systems::InputPort<double> &get_input_port_state() const {
    return this->get_input_port(state_input_port_);
  }

  const drake::systems::InputPort<double> &get_input_port_action() const {
    return this->get_input_port(action_input_port_);
  }

  const drake::systems::OutputPort<double> &get_output_port_next_state() const {
    return this->get_output_port(next_state_output_port_);
  }

private:
  void init(int n_x, int n_u, int n_lambda, int N, double dt);
  void SimulateOneStep(const drake::systems::Context<double> &context,
                       drake::systems::BasicVector<double> *next_state) const;

  drake::systems::InputPortIndex lcs_input_port_;
  drake::systems::InputPortIndex state_input_port_;
  drake::systems::InputPortIndex action_input_port_;
  drake::systems::OutputPortIndex next_state_output_port_;
};
} // namespace systems
} // namespace c3