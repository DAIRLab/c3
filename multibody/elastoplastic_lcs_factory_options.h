#pragma once

#include <optional>

#include "multibody/lcs_factory_options.h"

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

/** Configuration for an internal elastoplastic contact pair.  This can describe
 * the following kinds of behaviors:
 *  - Purely plastic:  yield_force is specified.
 *  - Purely elastic:  series spring/damper are specified.
 *  - Series elasto-plastic:  yield_force and series spring/damper are
 *      specified.
 *  - Parallel elasto-plastic:  yield_force and parallel spring/damper are
 *      specified.
 *  - Compound elasto-plastic:  yield_force, series spring/damper, and parallel
 *      spring/damper are all specified.
 */
struct ElastoPlasticContactPairConfig {
  std::string body_A;
  std::string body_B;
  std::string
      deformation_model;  // "plastic", "elastic", "series_elastoplastic",
                          // "parallel_elastoplastic", or
                          // "compound_elastoplastic"
  std::optional<double>
      yield_force;  // optional yield force for the contact pair, above
                    // which plastic deformation occurs
  std::optional<double>
      series_spring_constant;  // optional spring constant for a spring in
                               // series with the plastic joint
  std::optional<double>
      series_damper_constant;  // optional damper constant for a damper in
                               // series with the plastic joint
  std::optional<double>
      parallel_spring_constant;  // optional spring constant for a spring in
                                 // parallel with the plastic joint
  std::optional<double>
      parallel_damper_constant;  // optional damper constant for a damper in
                                 // parallel with the plastic joint

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(body_A));
    a->Visit(DRAKE_NVP(body_B));
    a->Visit(DRAKE_NVP(deformation_model));
    a->Visit(DRAKE_NVP(yield_force));
    a->Visit(DRAKE_NVP(series_spring_constant));
    a->Visit(DRAKE_NVP(series_damper_constant));
    a->Visit(DRAKE_NVP(parallel_spring_constant));
    a->Visit(DRAKE_NVP(parallel_damper_constant));

    ValidateContactPairConfig();
  }

 private:
  void ValidateContactPairConfig() const {
    DRAKE_DEMAND(!body_A.empty());
    DRAKE_DEMAND(!body_B.empty());

    if (deformation_model == "plastic") {
      DRAKE_DEMAND(yield_force.has_value());
    } else if (deformation_model == "elastic") {
      throw std::invalid_argument(
          "elastic deformation type is not currently supported");
      DRAKE_DEMAND(series_spring_constant.has_value() &&
                   series_damper_constant.has_value());
    } else if (deformation_model == "series_elastoplastic") {
      throw std::invalid_argument(
          "series elastoplastic deformation type is not currently supported");
      DRAKE_DEMAND(yield_force.has_value() &&
                   series_spring_constant.has_value() &&
                   series_damper_constant.has_value());
    } else if (deformation_model == "parallel_elastoplastic") {
      throw std::invalid_argument(
          "parallel elastoplastic deformation type is not currently supported");
      DRAKE_DEMAND(yield_force.has_value() &&
                   parallel_spring_constant.has_value() &&
                   parallel_damper_constant.has_value());
    } else if (deformation_model == "compound_elastoplastic") {
      throw std::invalid_argument(
          "compound elastoplastic deformation type is not currently supported");
      DRAKE_DEMAND(yield_force.has_value() &&
                   series_spring_constant.has_value() &&
                   series_damper_constant.has_value() &&
                   parallel_spring_constant.has_value() &&
                   parallel_damper_constant.has_value());
    } else {
      throw std::invalid_argument("invalid elastoplastic deformation type");
    }
  }
};

struct ElastoPlasticLCSFactoryOptions : LCSFactoryOptions {
  std::string deformation_model;  // Deformation model: "plastic", "elastic",
                                  // "series_elastoplastic",
                                  // "parallel_elastoplastic", or
                                  // "compound_elastoplastic"

  // Total number of internal contact points in the system
  std::optional<int> num_internal_contacts;

  // Detailed per-internal_contact-pair configurations (alternative to global
  // settings)
  std::optional<std::vector<ElastoPlasticContactPairConfig>>
      internal_contact_pair_configs;

  template <typename Archive>
  void Serialize(Archive* a) {
    LCSFactoryOptions::Serialize(a);
    a->Visit(DRAKE_NVP(deformation_model));
    a->Visit(DRAKE_NVP(num_internal_contacts));
    a->Visit(DRAKE_NVP(internal_contact_pair_configs));

    DRAKE_DEMAND(deformation_model ==
                 "plastic");  // TODO @bibit other deformation models currently
                              // unimplemented
  }

  // Returns the total number of external contacts.
  // Requires: num_contacts must be set and non-negative.
  // NOTE: ResolveNumContacts() from the base class still returns the number of
  // external contacts only, since that helper function is used for resolving
  // other external-only parameters, i.e. in ResolveNumFrictionDirections() and
  // ResolveMu().
  int ResolveNumExternalContacts() const { return ResolveNumContacts(); }

  // Returns the total number of internal contacts.
  // Requires: num_internal_contacts must be set and non-negative.
  int ResolveNumInternalContacts() const {
    DRAKE_DEMAND(num_internal_contacts.has_value());
    DRAKE_DEMAND(num_internal_contacts.value() >= 0);
    return num_internal_contacts.value();
  }

  void SetLCSFactoryOptionsFromBase(const LCSFactoryOptions& base_options) {
    contact_model = base_options.contact_model;
    N = base_options.N;
    dt = base_options.dt;
    if (base_options.num_contacts.has_value()) {
      num_contacts = base_options.num_contacts.value();
    }
    if (base_options.spring_stiffness.has_value()) {
      spring_stiffness = base_options.spring_stiffness;
    }
    if (base_options.contact_pair_configs.has_value()) {
      contact_pair_configs = base_options.contact_pair_configs;
    }
    if (base_options.num_friction_directions_per_contact.has_value()) {
      num_friction_directions_per_contact =
          base_options.num_friction_directions_per_contact;
    }
    if (base_options.mu_per_contact.has_value()) {
      mu_per_contact = base_options.mu_per_contact;
    }
    if (base_options.planar_normal_direction_per_contact.has_value()) {
      planar_normal_direction_per_contact =
          base_options.planar_normal_direction_per_contact;
    }
    if (base_options.num_friction_directions.has_value()) {
      num_friction_directions = base_options.num_friction_directions;
    }
    if (base_options.mu.has_value()) {
      mu = base_options.mu;
    }
    if (base_options.planar_normal_direction.has_value()) {
      planar_normal_direction = base_options.planar_normal_direction;
    }
  }
};

inline ElastoPlasticLCSFactoryOptions LoadElastoPlasticLCSFactoryOptions(
    const std::string& filename) {
  auto options =
      drake::yaml::LoadYamlFile<ElastoPlasticLCSFactoryOptions>(filename);
  return options;
}

}  // namespace c3
