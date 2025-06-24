#pragma once

#include "core/c3_options.h"
#include "multibody/lcs_factory_options.h"

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {
namespace systems {
/**
 * @struct C3StatePredictionJoint
 * @brief Represents the state prediction parameters for a joint in the system.
 *
 * This structure is used to define the properties of a joint, including its
 * name and the maximum allowable acceleration. The joints specified will be
 * used for clamping the predicted state during the optimization process to
 * ensure that the predicted states remain within physically realistic bounds
 * based on the maximum acceleration.
 *
 * In C3 Options file, the structure to specify these joints would look like:
 * ```
 * state_prediction_joints:
 *   - name: "joint_name"
 *     max_acceleration: 10.0  # Maximum acceleration in m/s^2
 *   - name: "another_joint_name"
 *     max_acceleration: 5.0   # Maximum acceleration in m/s^2
 * ```
 * or
 * ```
 * state_prediction_joints: []
 * ```
 * If no predicition is required
 */
struct C3StatePredictionJoint {
  // Joint name
  std::string name;
  // Maximum acceleration for the joint
  double max_acceleration;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(name));
    a->Visit(DRAKE_NVP(max_acceleration));
  }
};

struct C3ControllerOptions {
  std::string projection_type;  // "QP" or "MIQP"
  // C3 optimization options
  C3Options c3_options;

  // LCS (Linear Complementarity System) factory options
  LCSFactoryOptions lcs_factory_options;

  // Additional options specific to the C3Controller
  double solve_time_filter_alpha = 0.0;
  double publish_frequency = 100.0;  // Hz

  std::vector<C3StatePredictionJoint> state_prediction_joints;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(projection_type));
    a->Visit(DRAKE_NVP(c3_options));
    a->Visit(DRAKE_NVP(lcs_factory_options));
    a->Visit(DRAKE_NVP(solve_time_filter_alpha));
    a->Visit(DRAKE_NVP(publish_frequency));
    a->Visit(DRAKE_NVP(state_prediction_joints));

    if (projection_type == "QP") {
      DRAKE_DEMAND(lcs_factory_options.contact_model == "anitescu");
    }

    if (lcs_factory_options.contact_model == "frictionless_spring") {
      DRAKE_DEMAND(static_cast<int>(c3_options.g_lambda.size()) ==
                   lcs_factory_options.num_contacts);
      DRAKE_DEMAND(static_cast<int>(c3_options.u_lambda.size()) ==
                   lcs_factory_options.num_contacts);
    } else if (lcs_factory_options.contact_model == "stewart_and_trinkle") {
      DRAKE_DEMAND(static_cast<int>(c3_options.g_lambda.size()) ==
                   lcs_factory_options.num_contacts *
                           lcs_factory_options.num_friction_directions * 2 +
                       lcs_factory_options.num_contacts * 2);
      DRAKE_DEMAND(static_cast<int>(c3_options.u_lambda.size()) ==
                   lcs_factory_options.num_contacts *
                           lcs_factory_options.num_friction_directions * 2 +
                       lcs_factory_options.num_contacts * 2);
    } else if (lcs_factory_options.contact_model == "anitescu") {
      DRAKE_DEMAND(static_cast<int>(c3_options.g_lambda.size()) ==
                   lcs_factory_options.num_contacts *
                       lcs_factory_options.num_friction_directions * 2);
      DRAKE_DEMAND(static_cast<int>(c3_options.u_lambda.size()) ==
                   lcs_factory_options.num_contacts *
                       lcs_factory_options.num_friction_directions * 2);
    } else {
      throw std::runtime_error("unknown or unsupported contact model");
    }
  }
};

inline C3ControllerOptions LoadC3ControllerOptions(
    const std::string& filename) {
  auto options = drake::yaml::LoadYamlFile<C3ControllerOptions>(filename);
  return options;
}
}  // namespace systems
}  // namespace c3