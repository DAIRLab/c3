#pragma once

#include <optional>

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

struct ContactPairConfig {
  std::string body_A;
  std::string body_B;
  std::vector<int> body_A_collision_geom_indices;
  std::vector<int> body_B_collision_geom_indices;
  int num_friction_directions;
  std::optional<std::array<double, 3>>
      planar_normal_direction;  // optional normal vector for planar contact
  double mu;                    // friction coefficient

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(body_A));
    a->Visit(DRAKE_NVP(body_B));
    a->Visit(DRAKE_NVP(body_A_collision_geom_indices));
    a->Visit(DRAKE_NVP(body_B_collision_geom_indices));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(planar_normal_direction));
    a->Visit(DRAKE_NVP(mu));
  }
};

struct LCSFactoryOptions {
  std::string contact_model;  // "stewart_and_trinkle" or "anitescu" or
                              // "frictionless_spring"
  int num_contacts;           // Number of contacts in the system
  double spring_stiffness;    // Spring stiffness for frictionless spring model
  std::optional<int>
      num_friction_directions;  // Number of friction directions per contact
  std::optional<std::vector<int>>
      num_friction_directions_per_contact;  // Number of friction directions per
  std::optional<std::vector<double>>
      mu;  // Friction coefficient for each contact
  std::optional<std::array<double, 3>>
      planar_normal_direction;  // Optional normal vector for planar contact
  std::optional<std::vector<std::array<double, 3>>>
      planar_normal_direction_per_contact;  // Optional normal vector for planar
                                            // contact for each contact

  std::optional<std::vector<ContactPairConfig>>
      contact_pair_configs;  // Optional detailed contact pair configurations

  int N;      // number of time steps in the prediction horizon
  double dt;  // time step size

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(contact_model));
    a->Visit(DRAKE_NVP(num_contacts));
    a->Visit(DRAKE_NVP(spring_stiffness));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(num_friction_directions_per_contact));
    a->Visit(DRAKE_NVP(mu));
    a->Visit(DRAKE_NVP(planar_normal_direction));
    a->Visit(DRAKE_NVP(planar_normal_direction_per_contact));
    a->Visit(DRAKE_NVP(contact_pair_configs));

    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(dt));

    // Contact Pair Information contained in contact_pair_configs
    if (contact_pair_configs.has_value()) {
      DRAKE_DEMAND(contact_pair_configs.value().size() == (size_t)num_contacts);
    } else {
      // ensure mu and num_friction_directions are properly specified
      DRAKE_DEMAND(mu.has_value());
      DRAKE_DEMAND(mu.value().size() == (size_t)num_contacts);
      // if a single num_friction_directions is provided, use it for all
      // contacts
      if (num_friction_directions.has_value()) {
        num_friction_directions_per_contact =
            std::vector<int>(num_contacts, num_friction_directions.value());
      } else {
        DRAKE_DEMAND(num_friction_directions_per_contact.has_value());
        DRAKE_DEMAND(num_friction_directions_per_contact.value().size() ==
                     (size_t)num_contacts);
      }
    }
  }
};

inline LCSFactoryOptions LoadLCSFactoryOptions(const std::string& filename) {
  auto options = drake::yaml::LoadYamlFile<LCSFactoryOptions>(filename);
  return options;
}

}  // namespace c3