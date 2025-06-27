#include "multibody/multibody_utils.h"

#include <set>
#include <vector>

#include "drake/common/drake_assert.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/autodiff_gradient.h"

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::VectorX;
using drake::geometry::HalfSpace;
using drake::geometry::SceneGraph;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::multibody::BodyIndex;
using drake::multibody::JointActuatorIndex;
using drake::multibody::JointIndex;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::map;
using std::string;
using std::vector;

namespace c3 {
namespace multibody {

bool AreVectorsEqual(const Eigen::Ref<const AutoDiffVecXd>& a,
                     const Eigen::Ref<const AutoDiffVecXd>& b) {
  if (a.rows() != b.rows()) {
    return false;
  }
  if (ExtractValue(a) != ExtractValue(b)) {
    return false;
  }
  const Eigen::MatrixXd a_gradient = ExtractGradient(a);
  const Eigen::MatrixXd b_gradient = ExtractGradient(b);
  if (a_gradient.rows() != b_gradient.rows() ||
      a_gradient.cols() != b_gradient.cols()) {
    return false;
  }
  return a_gradient == b_gradient;
}

bool AreVectorsEqual(const Eigen::Ref<const VectorXd>& a,
                     const Eigen::Ref<const VectorXd>& b) {
  return a == b;
}

template <typename T>
VectorX<T> GetInput(const MultibodyPlant<T>& plant, const Context<T>& context) {
  if (plant.num_actuators() > 0) {
    VectorX<T> input = plant.get_actuation_input_port().Eval(context);
    return input;
  } else {
    return VectorX<T>(0);
  }
}


template <typename T>
void SetContext(const MultibodyPlant<T>& plant,
                const Eigen::Ref<const VectorX<T>>& state,
                const Eigen::Ref<const VectorX<T>>& input,
                Context<T>* context) {
  SetPositionsIfNew<T>(plant, state.head(plant.num_positions()), context);
  SetVelocitiesIfNew<T>(plant, state.tail(plant.num_velocities()), context);
  SetInputsIfNew<T>(plant, input, context);
}

template <typename T>
void SetPositionsAndVelocitiesIfNew(const MultibodyPlant<T>& plant,
                                    const Eigen::Ref<const VectorX<T>>& x,
                                    Context<T>* context) {
  SetPositionsIfNew<T>(plant, x.head(plant.num_positions()), context);
  SetVelocitiesIfNew<T>(plant, x.tail(plant.num_velocities()), context);
}

template <typename T>
void SetPositionsIfNew(const MultibodyPlant<T>& plant,
                       const Eigen::Ref<const VectorX<T>>& q,
                       Context<T>* context) {
  if (!AreVectorsEqual(q, plant.GetPositions(*context))) {
    plant.SetPositions(context, q);
  }
}

template <typename T>
void SetVelocitiesIfNew(const MultibodyPlant<T>& plant,
                        const Eigen::Ref<const VectorX<T>>& v,
                        Context<T>* context) {
  if (!AreVectorsEqual(v, plant.GetVelocities(*context))) {
    plant.SetVelocities(context, v);
  }
}

template <typename T>
void SetInputsIfNew(const MultibodyPlant<T>& plant,
                    const Eigen::Ref<const VectorX<T>>& u,
                    Context<T>* context) {
  if (!plant.get_actuation_input_port().HasValue(*context) ||
      !AreVectorsEqual(u, plant.get_actuation_input_port().Eval(*context))) {
    plant.get_actuation_input_port().FixValue(context, u);
  }
}


template void SetContext(const MultibodyPlant<double>& plant,
                         const Eigen::Ref<const VectorXd>& state,
                         const Eigen::Ref<const VectorXd>&,
                         Context<double>* context);  // NOLINT
template void SetContext(const MultibodyPlant<AutoDiffXd>& plant,
                         const Eigen::Ref<const AutoDiffVecXd>& state,
                         const Eigen::Ref<const AutoDiffVecXd>&,
                         Context<AutoDiffXd>* context);  // NOLINT
template void SetPositionsAndVelocitiesIfNew(
    const MultibodyPlant<AutoDiffXd>&, const Eigen::Ref<const AutoDiffVecXd>&,
    Context<AutoDiffXd>*);  // NOLINT
template void SetPositionsAndVelocitiesIfNew(const MultibodyPlant<double>&,
                                             const Eigen::Ref<const VectorXd>&,
                                             Context<double>*);  // NOLINT
template void SetPositionsIfNew(const MultibodyPlant<AutoDiffXd>&,
                                const Eigen::Ref<const AutoDiffVecXd>&,
                                Context<AutoDiffXd>*);  // NOLINT
template void SetPositionsIfNew(const MultibodyPlant<double>&,
                                const Eigen::Ref<const VectorXd>&,
                                Context<double>*);  // NOLINT
template void SetVelocitiesIfNew(const MultibodyPlant<AutoDiffXd>&,
                                 const Eigen::Ref<const AutoDiffVecXd>&,
                                 Context<AutoDiffXd>*);  // NOLINT
template void SetVelocitiesIfNew(const MultibodyPlant<double>&,
                                 const Eigen::Ref<const VectorXd>&,
                                 Context<double>*);  // NOLINT
template void SetInputsIfNew(const MultibodyPlant<AutoDiffXd>&,
                             const Eigen::Ref<const AutoDiffVecXd>&,
                             Context<AutoDiffXd>*);  // NOLINT
template void SetInputsIfNew(const MultibodyPlant<double>&,
                             const Eigen::Ref<const VectorXd>&,
                             Context<double>*);  // NOLINT
}  // namespace multibody
}  // namespace c3
