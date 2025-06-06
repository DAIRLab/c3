#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>

#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

using drake::geometry::SceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;
using Eigen::VectorXd;

namespace c3 {
namespace systems {
namespace test {

// A system that integrates a Cartpole model with Drake's SceneGraph for
// visualization.
class CartpoleWithSoftWallsVisualizer
    : public drake::systems::LeafSystem<double> {
 public:
  // Constructor: Initializes the MultibodyPlant and connects it to the
  // SceneGraph.
  CartpoleWithSoftWallsVisualizer(SceneGraph<double>* scene_graph,
                                  double time_step = 0.0) {
    plant = std::make_unique<MultibodyPlant<double>>(time_step);
    Parser parser(plant.get(), scene_graph);

    // Load the Cartpole model from an SDF file.
    const std::string file = "systems/test/res/cartpole_softwalls.sdf";
    parser.AddModels(file);
    plant->Finalize();

    // Declare input and output ports.
    state_input_port_ = this->DeclareVectorInputPort(
                                "state", drake::systems::BasicVector<double>(
                                             plant->num_multibody_states()))
                            .get_index();
    cartpole_position_output_port_ =
        this->DeclareVectorOutputPort(
                "cartpole_position", 2,
                &CartpoleWithSoftWallsVisualizer::CalcCartpolePosition)
            .get_index();  // cartpole state is 2D (x, theta)
  }

  // Returns the input port for the Cartpole state.
  const drake::systems::InputPort<double>& get_state_input_port() const {
    return this->get_input_port(state_input_port_);
  }

  // Returns the output port for the SceneGraph state.
  const drake::systems::OutputPort<double>& get_cartpole_position_output_port()
      const {
    return this->get_output_port(cartpole_position_output_port_);
  }

  // Adds the CartpoleWithSoftWallsVisualizer system to a DiagramBuilder and
  // connects it to the SceneGraph.
  static CartpoleWithSoftWallsVisualizer* AddToBuilder(
      drake::systems::DiagramBuilder<double>* builder,
      SceneGraph<double>* scene_graph,
      const drake::systems::OutputPort<double>& state_port,
      double time_step = 0.0) {
    DRAKE_DEMAND(builder != nullptr && scene_graph != nullptr);
    // Create the CartpoleWithSoftWallsVisualizer system.
    auto geometry = builder->AddSystem<CartpoleWithSoftWallsVisualizer>(
        scene_graph, time_step);

    // Connect the state port to the input port of the
    // CartpoleWithSoftWallsVisualizer system.
    builder->Connect(state_port, geometry->get_state_input_port());

    // Add a MultibodyPositionToGeometryPose system to convert state to
    // geometry pose.
    auto to_geometry_pose =
        builder->AddSystem<MultibodyPositionToGeometryPose<double>>(
            *(geometry->plant));

    // Connect the output ports to the SceneGraph.
    builder->Connect(geometry->get_cartpole_position_output_port(),
                     to_geometry_pose->get_input_port());
    builder->Connect(to_geometry_pose->get_output_port(),
                     scene_graph->get_source_pose_port(
                         geometry->plant->get_source_id().value()));

    return geometry;
  }

  std::unique_ptr<MultibodyPlant<double>>
      plant;  // Pointer to the MultibodyPlant.

 private:
  // Gets the cartpole position given the state
  void CalcCartpolePosition(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* cartpole_position) const {
    // Extract the state vector and map it to a 2D vector.
    auto state = this->EvalVectorInput(context, state_input_port_)->value();
    VectorXd x = Eigen::Map<VectorXd>(state.data(), 2);

    // Set the output vector for the SceneGraph.
    cartpole_position->SetFromVector(x);
  }

  drake::systems::InputPortIndex
      state_input_port_;  // Input port index for state.
  drake::systems::OutputPortIndex
      cartpole_position_output_port_;  // Output port index for SceneGraph
                                       // state.
};

}  // namespace test
}  // namespace systems
}  // namespace c3