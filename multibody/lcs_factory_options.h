#pragma once

#include <optional>

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

struct ContactPairConfig {
  std::string body_A;
  std::string body_B;
  std::optional<std::vector<int>> body_A_collision_geom_indices;
  std::optional<std::vector<int>> body_B_collision_geom_indices;
  int num_friction_directions;
  double mu;  // friction coefficient
  std::optional<std::array<double, 3>>
      planar_normal_direction;  // optional normal vector for planar contact
  std::optional<unsigned int>
      num_active_contact_pairs;  // Number of geometry-geometry contact pairs to
                                 // consider (e.g., if body_A has 10 geoms and
                                 // body_B has 5 geoms, there are 50 possible
                                 // pairs. Set this to limit how many are used
                                 // in the LCS. If unspecified, all pairs are
                                 // considered.)

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(body_A));
    a->Visit(DRAKE_NVP(body_B));
    a->Visit(DRAKE_NVP(body_A_collision_geom_indices));
    a->Visit(DRAKE_NVP(body_B_collision_geom_indices));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(planar_normal_direction));
    a->Visit(DRAKE_NVP(mu));
    a->Visit(DRAKE_NVP(num_active_contact_pairs));

    ValidateContactPairConfig();
  }

 private:
  void ValidateContactPairConfig() const {
    DRAKE_DEMAND(!body_A.empty());
    DRAKE_DEMAND(!body_B.empty());
    if (body_A_collision_geom_indices.has_value()) {
      for (int idx : body_A_collision_geom_indices.value()) {
        DRAKE_DEMAND(idx >= 0);
      }
    }
    if (body_B_collision_geom_indices.has_value()) {
      for (int idx : body_B_collision_geom_indices.value()) {
        DRAKE_DEMAND(idx >= 0);
      }
    }
    DRAKE_DEMAND(num_friction_directions >= 0);
    DRAKE_DEMAND(mu >= 0);  // Friction coefficient must be non-negative
    if (num_friction_directions == 1) {
      DRAKE_DEMAND(
          planar_normal_direction.has_value());  // For planar contact, the
                                                 // normal direction must be
                                                 // specified
      const auto& normal = planar_normal_direction.value();
      double norm = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                              normal[2] * normal[2]);
      DRAKE_DEMAND(std::abs(norm - 1) < 1e-6);  // Ensure it's a unit vector
    }

    if (num_active_contact_pairs.has_value()) {
      DRAKE_DEMAND(num_active_contact_pairs.value() > 0);
    }
  }
};

struct LCSFactoryOptions {
  // Contact model type: "stewart_and_trinkle", "anitescu", or
  // "frictionless_spring"
  std::string contact_model;
  int N;      // Number of time steps in the prediction horizon
  double dt;  // Time step size (seconds)

  // Total number of contact points in the system
  std::optional<int> num_contacts;

  // Spring stiffness for frictionless_spring contact model (must be positive)
  std::optional<double> spring_stiffness;

  // Detailed per-contact-pair configurations (alternative to global settings)
  std::optional<std::vector<ContactPairConfig>> contact_pair_configs;

  // Per-contact friction parameters (must match num_contacts if provided)
  std::optional<std::vector<int>> num_friction_directions_per_contact;
  std::optional<std::vector<double>> mu_per_contact;
  std::optional<std::vector<std::array<double, 3>>>
      planar_normal_direction_per_contact;

  // Global friction parameters (applied to all contacts if per-contact not
  // specified)
  std::optional<int> num_friction_directions;
  std::optional<double> mu;
  std::optional<std::array<double, 3>> planar_normal_direction;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(contact_model));
    a->Visit(DRAKE_NVP(num_contacts));
    a->Visit(DRAKE_NVP(spring_stiffness));
    a->Visit(DRAKE_NVP(contact_pair_configs));
    a->Visit(DRAKE_NVP(num_friction_directions_per_contact));
    a->Visit(DRAKE_NVP(mu_per_contact));
    a->Visit(DRAKE_NVP(planar_normal_direction_per_contact));
    a->Visit(DRAKE_NVP(num_friction_directions));
    a->Visit(DRAKE_NVP(mu));
    a->Visit(DRAKE_NVP(planar_normal_direction));
    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(dt));

    if (contact_model == "frictionless_spring") {
      DRAKE_DEMAND(spring_stiffness.has_value());
      DRAKE_DEMAND(spring_stiffness.value() > 0);
    }
  }

  // Returns the total number of contact points.
  // Requires: num_contacts must be set and non-negative.
  int ResolveNumContacts() const {
    DRAKE_DEMAND(num_contacts.has_value());
    DRAKE_DEMAND(num_contacts.value() >= 0);
    return num_contacts.value();
  }

  // Resolves the number of friction directions for each contact point.
  // Returns a vector of length num_contacts where each element specifies
  // the friction direction count for that contact.
  // Uses per-contact values if available, otherwise broadcasts the global
  // value to all contacts.
  // Requires: num_contacts and either num_friction_directions_per_contact
  //           or num_friction_directions must be set.
  std::vector<int> ResolveNumFrictionDirections() const {
    if (contact_model == "frictionless_spring")
      return std::vector<int>(ResolveNumContacts(), 0);

    DRAKE_DEMAND(num_friction_directions_per_contact.has_value() ||
                 num_friction_directions.has_value());

    if (num_friction_directions_per_contact.has_value()) {
      DRAKE_DEMAND(num_friction_directions_per_contact.value().size() ==
                   (size_t)num_contacts.value());
      for (int n : num_friction_directions_per_contact.value()) {
        DRAKE_DEMAND(n >= 0);
      }
      return num_friction_directions_per_contact.value();
    }

    return std::vector<int>(ResolveNumContacts(),
                            num_friction_directions.value());
  }

  // Resolves the friction coefficient (mu) for each contact point.
  // Returns a vector of length num_contacts where each element specifies
  // the friction coefficient for that contact.
  // Uses per-contact values if available, otherwise broadcasts the global
  // value to all contacts.
  // Requires: num_contacts and either mu_per_contact or mu must be set.
  std::vector<double> ResolveMu() const {
    if (contact_model == "frictionless_spring")
      return std::vector<double>(ResolveNumContacts(), 0.0);

    DRAKE_DEMAND(mu_per_contact.has_value() || mu.has_value());

    if (mu_per_contact.has_value()) {
      DRAKE_DEMAND(mu_per_contact.value().size() ==
                   (size_t)num_contacts.value());
      for (double coeff : mu_per_contact.value()) {
        DRAKE_DEMAND(coeff >= 0);
      }
      return mu_per_contact.value();
    }

    return std::vector<double>(ResolveNumContacts(), mu.value());
  }

  // Resolves the planar normal direction for each contact point.
  // Returns a vector of length num_contacts where each element is either:
  //   - A unit normal vector for planar contacts (num_friction_directions==1)
  //   - std::nullopt for non-planar contacts (num_friction_directions!=1)
  // Uses per-contact normals if available, otherwise broadcasts the global
  // normal to all planar contacts.
  // Requires: num_contacts must be set, and planar contacts must have
  //           either planar_normal_direction_per_contact or
  //           planar_normal_direction specified.
  std::vector<std::optional<std::array<double, 3>>> ResolvePlanarNormals()
      const {
    DRAKE_DEMAND(num_contacts.has_value());
    DRAKE_DEMAND(num_contacts.value() >= 0);

    auto n_friction_direction_per_contact = ResolveNumFrictionDirections();
    DRAKE_DEMAND(n_friction_direction_per_contact.size() ==
                 (size_t)num_contacts.value());

    std::vector<std::optional<std::array<double, 3>>> resolved_normals;
    for (size_t i = 0; i < n_friction_direction_per_contact.size(); ++i) {
      if (contact_model == "frictionless_spring" ||
          n_friction_direction_per_contact[i] == 1) {
        if (planar_normal_direction_per_contact.has_value()) {
          DRAKE_DEMAND(planar_normal_direction_per_contact.value().size() ==
                       (size_t)num_contacts.value());
          resolved_normals.push_back(
              planar_normal_direction_per_contact.value()[i]);
        } else {
          DRAKE_DEMAND(planar_normal_direction.has_value());
          resolved_normals.push_back(planar_normal_direction);
        }
      } else {
        resolved_normals.push_back(std::nullopt);
      }
    }
    return resolved_normals;
  }
};

inline LCSFactoryOptions LoadLCSFactoryOptions(const std::string& filename) {
  auto options = drake::yaml::LoadYamlFile<LCSFactoryOptions>(filename);
  return options;
}

}  // namespace c3
