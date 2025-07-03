#pragma once

#include <string>
#include <vector>

namespace c3 {
namespace systems {
namespace lcmt_generators {

struct TrajectoryDescription {
  std::string trajectory_name;  // Name of the trajectory
  std::string variable_type;  // Type of C3 variable (e.g., "x", "u", "lambda")
  std::vector<std::tuple<int, int>> indices;
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(trajectory_name));
    a->Visit(DRAKE_NVP(variable_type));
    a->Visit(DRAKE_NVP(indices));
  }
};

struct TrajectoryGeneratorOptions {
  std::vector<TrajectoryDescription>
      trajectories;  // List of trajectories to generate
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(trajectories));
  }
};

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3