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
namespace test {

// A system that integrates a Cartpole model with Drake's SceneGraph for
// visualization.
class CartpoleGeometry : public drake::systems::LeafSystem<double> {
 public:
  // Constructor: Initializes the MultibodyPlant and connects it to the
  // SceneGraph.
  CartpoleGeometry(SceneGraph<double>* scene_graph, double time_step = 0.0) {
    plant = new MultibodyPlant<double>(time_step);
    Parser parser(plant, scene_graph);

    // Load the Cartpole model from an SDF file.
    const std::string file = "systems/test/res/cartpole_softwalls.sdf";
    parser.AddModels(file);
    plant->Finalize();

    // Declare input and output ports.
    state_input_port_ = this->DeclareVectorInputPort(
                                "state", drake::systems::BasicVector<double>(4))
                            .get_index();
    scene_graph_state_output_port_ =
        this->DeclareVectorOutputPort("scene_graph_state", 2,
                                      &CartpoleGeometry::OutputGeometryPose)
            .get_index();  // cartpole state is 2D (x, theta)
  }

  // Returns the input port for the Cartpole state.
  const drake::systems::InputPort<double>& get_input_port_state() const {
    return this->get_input_port(state_input_port_);
  }

  // Returns the output port for the SceneGraph state.
  const drake::systems::OutputPort<double>& get_output_port_scene_graph_state()
      const {
    return this->get_output_port(scene_graph_state_output_port_);
  }

  // Adds the CartpoleGeometry system to a DiagramBuilder and connects it to the
  // SceneGraph.
  static CartpoleGeometry* AddToBuilder(
      drake::systems::DiagramBuilder<double>* builder,
      SceneGraph<double>* scene_graph,
      const drake::systems::OutputPort<double>& state_port,
      double time_step = 0.0) {
    // Create the CartpoleGeometry system.
    CartpoleGeometry* geometry =
        builder->AddSystem<CartpoleGeometry>(scene_graph, time_step);

    // Connect the state port to the input port of the CartpoleGeometry system.
    builder->Connect(state_port, geometry->get_input_port_state());

    // Add a MultibodyPositionToGeometryPose system to convert state to geometry
    // pose.
    auto to_geometry_pose =
        builder->AddSystem<MultibodyPositionToGeometryPose<double>>(
            *(geometry->plant));

    // Connect the output ports to the SceneGraph.
    builder->Connect(geometry->get_output_port_scene_graph_state(),
                     to_geometry_pose->get_input_port());
    builder->Connect(to_geometry_pose->get_output_port(),
                     scene_graph->get_source_pose_port(
                         geometry->plant->get_source_id().value()));

    return geometry;
  }

  MultibodyPlant<double>* plant;  // Pointer to the MultibodyPlant.

 private:
  // Outputs the geometry pose for the SceneGraph based on the Cartpole state.
  void OutputGeometryPose(
      const drake::systems::Context<double>& context,
      drake::systems::BasicVector<double>* scene_graph_state) const {
    // Extract the state vector and map it to a 2D vector.
    auto state = this->EvalVectorInput(context, state_input_port_)->value();
    VectorXd x = Eigen::Map<VectorXd>(state.data(), 2);

    // Set the output vector for the SceneGraph.
    scene_graph_state->SetFromVector(x);
  }

  drake::systems::InputPortIndex
      state_input_port_;  // Input port index for state.
  drake::systems::OutputPortIndex
      scene_graph_state_output_port_;  // Output port index for SceneGraph
                                       // state.
};

}  // namespace test
}  // namespace c3