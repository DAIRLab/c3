#pragma once

#include <string>
#include <vector>

namespace c3 {
namespace systems {
namespace lcmt_generators {

struct TrajectoryDescription {
  // ===========================================================================
  // Index Range for Extracting Vector Slices
  // ===========================================================================
  //
  // The index_range struct defines a contiguous slice of a vector (state,
  // input, or force) to extract and publish as a separate trajectory.
  //
  // Purpose:
  // --------
  // C3 systems typically have large state/input/force vectors. Often, we only
  // want to visualize or log a subset of these variables. For example:
  //   - Extract positions [0:6] from a 14-dimensional state vector
  //   - Extract a single actuator force [3] from a 10-dimensional input vector
  //   - Extract contact forces [0:3] from a 20-dimensional force vector
  //
  // Usage:
  // ------
  // Given a vector x of size n, index_range {start, end} extracts:
  //   x[start], x[start+1], ..., x[end-1]
  //
  // This follows standard C++ iterator conventions where:
  //   - start: inclusive (first element to include)
  //   - end: exclusive (one past the last element to include)
  //
  // Examples:
  // ---------
  // 1. Extract full 7-DOF robot joint positions:
  //    {start: 0, end: 7}  -> extracts indices [0, 1, 2, 3, 4, 5, 6]
  //
  // 2. Extract x, y, z position from state:
  //    {start: 0, end: 3}  -> extracts indices [0, 1, 2]
  //
  // 3. Extract single element:
  //    {start: 5, end: 6}  -> extracts index [5]
  //
  // Multiple Ranges:
  // ----------------
  // The 'indices' vector allows specifying multiple non-contiguous ranges
  // that will be concatenated into a single trajectory. For example:
  //   indices: [{start: 0, end: 3}, {start: 7, end: 10}]
  //   -> extracts [0, 1, 2, 7, 8, 9] and concatenates them
  //
  // This is useful when you want to publish related but non-contiguous
  // variables together, such as positions from multiple joints that aren't
  // adjacent in the state vector.
  // ===========================================================================
  struct index_range {
    int start;  // Start index (inclusive) - first element to extract
    int end;    // End index (exclusive) - one past the last element to extract

    template <typename Archive>
    void Serialize(Archive* a) {
      a->Visit(DRAKE_NVP(start));
      a->Visit(DRAKE_NVP(end));
    }
  };

  std::string trajectory_name;  // Name of the trajectory for LCM publishing
  std::string variable_type;    // Type of C3 variable: "state", "input", or "force"
  std::vector<index_range> indices;  // List of index ranges to extract and concatenate

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
      trajectories;  // List of trajectories to generate and publish via LCM

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(trajectories));
  }
};

}  // namespace lcmt_generators
}  // namespace systems
}  // namespace c3