#pragma once

#include <string>
#include <vector>

namespace c3 {
namespace systems {
namespace lcmt_generators {

struct TrajectoryDescription {
  struct index_range {
    int start;  // Start index of the range
    int end;    // End index of the range
    template <typename Archive>
    void Serialize(Archive* a) {
      a->Visit(DRAKE_NVP(start));
      a->Visit(DRAKE_NVP(end));
    }
  };
  std::string trajectory_name;  // Name of the trajectory
  std::string variable_type;  // Type of C3 variable ["state", "input", "force"]
  std::vector<index_range> indices;
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(trajectory_name));
    a->Visit(DRAKE_NVP(variable_type));

    DRAKE_ASSERT(variable_type == "state" || variable_type == "input" ||
                 variable_type == "force");

    a->Visit(DRAKE_NVP(indices));
  }
};

struct C3TrajectoryGeneratorConfig {
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