#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>

using drake::geometry::SceneGraph;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::systems::rendering::MultibodyPositionToGeometryPose;
using Eigen::VectorXd;

namespace c3 {
namespace test {
class CartpoleGeometry : public drake::systems::LeafSystem<double> {
public:
  CartpoleGeometry(SceneGraph<double> *scene_graph, double time_step = 0.0) {
    plant = new MultibodyPlant<double>(time_step);
    Parser parser(plant, scene_graph);

    const std::string file = "core/controllers/test/sdf/cartpole_softwalls.sdf";
    parser.AddModels(file);
    plant->Finalize();

    state_input_port_ = this->DeclareVectorInputPort(
                                "state", drake::systems::BasicVector<double>(4))
                            .get_index();
    scene_graph_state_output_port_ =
        this->DeclareVectorOutputPort("scene_graph_state", 2,
                                      &CartpoleGeometry::OutputGeometryPose)
            .get_index();
  }

  const drake::systems::InputPort<double> &get_input_port_state() const {
    return this->get_input_port(state_input_port_);
  }
  const drake::systems::OutputPort<double> &
  get_output_port_scene_graph_state() const {
    return this->get_output_port(scene_graph_state_output_port_);
  }

  static CartpoleGeometry *
  AddToBuilder(drake::systems::DiagramBuilder<double> *builder,
               SceneGraph<double> *scene_graph,
               const drake::systems::OutputPort<double> &state_port,
               double time_step = 0.0) {
    // CartpoleGeometry *geometry =
    //     builder->AddSystem<CartpoleGeometry>(scene_graph, time_step);
    // builder->Connect(state_port, geometry->get_input_port_state());
    // auto to_geometry_pose =
    //     builder->AddSystem<MultibodyPositionToGeometryPose<double>>(
    //         *(geometry->plant));

    // builder->Connect(geometry->get_output_port_scene_graph_state(),
    //                  to_geometry_pose->get_input_port());
    // builder->Connect(to_geometry_pose->get_output_port(),
    //                  scene_graph->get_source_pose_port(
    //                      geometry->plant->get_source_id().value()));

    return nullptr;
  }

  MultibodyPlant<double> *plant;

private:
  void OutputGeometryPose(
      const drake::systems::Context<double> &context,
      drake::systems::BasicVector<double> *scene_graph_state) const {

    auto state = this->EvalVectorInput(context, state_input_port_)->value();
    VectorXd x = Eigen::Map<VectorXd>(state.data(), 2);
    scene_graph_state->SetFromVector(x);
  }
  drake::systems::InputPortIndex state_input_port_;
  drake::systems::OutputPortIndex scene_graph_state_output_port_;
};
} // namespace test
} // namespace c3