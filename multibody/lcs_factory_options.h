#pragma once

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

struct LCSFactoryOptions {
  std::string contact_model;    // "stewart_and_trinkle" or "anitescu" or
                                // "frictionless_spring"
  int num_friction_directions;  // Number of friction directions per contact
  int num_contacts;             // Number of contacts in the system
  double spring_stiffness;  // Spring stiffness for frictionless spring model
  std::vector<double> mu;   // Friction coefficient for each contact

  int N;      // number of time steps in the prediction horizon
  double dt;  // time step size

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(contact_model));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(num_contacts));
    a->Visit(DRAKE_NVP(spring_stiffness));
    a->Visit(DRAKE_NVP(mu));

    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(dt));

    DRAKE_DEMAND(mu.size() == (size_t)num_contacts);
  }
};

inline LCSFactoryOptions LoadLCSFactoryOptions(const std::string& filename) {
  auto options = drake::yaml::LoadYamlFile<LCSFactoryOptions>(filename);
  return options;
}

}  // namespace c3