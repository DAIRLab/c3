#include "systems/c3_controller.h"

#include "drake/systems/framework/leaf_system.h"

using c3::systems::C3Output;

// Custom LeafSystem to process C3 solutions and to get input values.
class C3Solution2Input : public drake::systems::LeafSystem<double> {
 public:
  explicit C3Solution2Input(int n_u) : n_u_(n_u) {
    // Declare input port for C3 solutions.
    c3_solution_port_index_ =
        this->DeclareAbstractInputPort("c3_solution",
                                       drake::Value<C3Output::C3Solution>())
            .get_index();
    // Declare output port for inputs.
    c3_input_port_index_ =
        this->DeclareVectorOutputPort("u", n_u_, &C3Solution2Input::GetC3Input)
            .get_index();
  }

  // Getter for the input port.
  const drake::systems::InputPort<double>& get_input_port_c3_solution() const {
    return this->get_input_port(c3_solution_port_index_);
  }

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_output_port_c3_input() const {
    return this->get_output_port(c3_input_port_index_);
  }

 private:
  drake::systems::InputPortIndex c3_solution_port_index_;
  drake::systems::OutputPortIndex c3_input_port_index_;

  int n_u_;

  // Compute the input from the C3 solution.
  void GetC3Input(const drake::systems::Context<double>& context,
                  drake::systems::BasicVector<double>* output) const {
    const drake::AbstractValue* input = this->EvalAbstractInput(context, 0);
    DRAKE_ASSERT(input != nullptr);
    const auto& sol = input->get_value<C3Output::C3Solution>();
    output->get_mutable_value() = sol.u_sol_.col(0).cast<double>();
  }
};

// Converts a vector to a timestamped vector.
class Vector2TimestampedVector : public drake::systems::LeafSystem<double> {
 public:
  explicit Vector2TimestampedVector(int n) {
    vector_port_index_ = this->DeclareVectorInputPort("state", n).get_index();
    timestamped_vector_port_index_ =
        this->DeclareVectorOutputPort("timestamped_state",
                                      c3::systems::TimestampedVector<double>(n),
                                      &Vector2TimestampedVector::Convert)
            .get_index();
  }

  // Getter for the input port.
  const drake::systems::InputPort<double>& get_input_port_state() const {
    return this->get_input_port(vector_port_index_);
  }

  // Getter for the output port.
  const drake::systems::OutputPort<double>& get_output_port_timestamped_state()
      const {
    return this->get_output_port(timestamped_vector_port_index_);
  }

 private:
  drake::systems::InputPortIndex vector_port_index_;
  drake::systems::OutputPortIndex timestamped_vector_port_index_;

  // Convert input vector to timestamped vector.
  void Convert(const drake::systems::Context<double>& context,
               c3::systems::TimestampedVector<double>* output) const {
    output->SetDataVector(
        this->EvalVectorInput(context, vector_port_index_)->get_value());
    output->set_timestamp(context.get_time());  // Set timestamp.
  }
};