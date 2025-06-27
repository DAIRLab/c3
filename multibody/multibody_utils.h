#pragma once

#include <map>
#include <string>

#include "drake/multibody/plant/multibody_plant.h"

namespace c3 {
namespace multibody {

/// Update an existing MultibodyPlant context, setting corresponding state and
/// input values.
/// Will only set values that have changed.
template <typename T>
void SetContext(const drake::multibody::MultibodyPlant<T>& plant,
                const Eigen::Ref<const drake::VectorX<T>>& state,
                const Eigen::Ref<const drake::VectorX<T>>& input,
                drake::systems::Context<T>* context);

/// Update an existing MultibodyPlant context, setting corresponding positions.
/// Will only set if value if changed from current value.
template <typename T>
void SetPositionsAndVelocitiesIfNew(
    const drake::multibody::MultibodyPlant<T>& plant,
    const Eigen::Ref<const drake::VectorX<T>>& x,
    drake::systems::Context<T>* context);

/// Update an existing MultibodyPlant context, setting corresponding positions.
/// Will only set if value if changed from current value.
template <typename T>
void SetPositionsIfNew(const drake::multibody::MultibodyPlant<T>& plant,
                       const Eigen::Ref<const drake::VectorX<T>>& q,
                       drake::systems::Context<T>* context);

/// Update an existing MultibodyPlant context, setting corresponding velocities.
/// Will only set if value if changed from current value.
template <typename T>
void SetVelocitiesIfNew(const drake::multibody::MultibodyPlant<T>& plant,
                        const Eigen::Ref<const drake::VectorX<T>>& f,
                        drake::systems::Context<T>* context);

/// Update an existing MultibodyPlant context, setting corresponding inputs.
/// Will only set if value if changed from current value.
template <typename T>
void SetInputsIfNew(const drake::multibody::MultibodyPlant<T>& plant,
                    const Eigen::Ref<const drake::VectorX<T>>& u,
                    drake::systems::Context<T>* context);

}  // namespace multibody
}  // namespace c3
