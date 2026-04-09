#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "core/lcs.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"
#include "multibody/lcs_factory_options.h"
#include "multibody/multibody_utils.h"

#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"

namespace py = pybind11;
namespace c3 {
namespace multibody {
namespace pyc3 {
PYBIND11_MODULE(multibody, m) {
  m.doc() = "C3 Multibody Utilities";

  // LCSFactory Class and ContactModel enum
  py::enum_<c3::multibody::ContactModel>(m, "ContactModel")
      .value("Unknown", c3::multibody::ContactModel::kUnknown)
      .value("StewartAndTrinkle",
             c3::multibody::ContactModel::kStewartAndTrinkle)
      .value("Anitescu", c3::multibody::ContactModel::kAnitescu)
      .value("FrictionlessSpring",
             c3::multibody::ContactModel::kFrictionlessSpring)
      .export_values();

  py::class_<c3::multibody::LCSContactDescription>(m, "LCSContactDescription")
      .def(py::init<>())
      .def_readwrite("witness_point_A",
                     &c3::multibody::LCSContactDescription::witness_point_A)
      .def_readwrite("witness_point_B",
                     &c3::multibody::LCSContactDescription::witness_point_B)
      .def_readwrite("force_basis",
                     &c3::multibody::LCSContactDescription::force_basis)
      .def_readwrite("is_slack",
                     &c3::multibody::LCSContactDescription::is_slack)
      .def_static("CreateSlackVariableDescription",
                  &c3::multibody::LCSContactDescription::
                      CreateSlackVariableDescription);

  py::class_<ContactPairConfig>(m, "ContactPairConfig")
      .def(py::init<>())
      .def_readwrite("body_A", &ContactPairConfig::body_A)
      .def_readwrite("body_B", &ContactPairConfig::body_B)
      .def_readwrite("body_A_collision_geom_indices",
                     &ContactPairConfig::body_A_collision_geom_indices)
      .def_readwrite("body_B_collision_geom_indices",
                     &ContactPairConfig::body_B_collision_geom_indices)
      .def_readwrite("mu", &ContactPairConfig::mu)
      .def_readwrite("num_friction_directions",
                     &ContactPairConfig::num_friction_directions)
      .def_readwrite("planar_normal_direction",
                     &ContactPairConfig::planar_normal_direction)
      .def_readwrite("num_active_contact_pairs",
                     &ContactPairConfig::num_active_contact_pairs);

  py::class_<LCSFactoryOptions>(m, "LCSFactoryOptions")
      .def(py::init<>())
      .def_readwrite("dt", &LCSFactoryOptions::dt)
      .def_readwrite("N", &LCSFactoryOptions::N)
      .def_property(
          "contact_model",
          [](const LCSFactoryOptions& self) {
            // Convert string back to enum for Python
            if (self.contact_model == "stewart_and_trinkle")
              return c3::multibody::ContactModel::kStewartAndTrinkle;
            if (self.contact_model == "anitescu")
              return c3::multibody::ContactModel::kAnitescu;
            if (self.contact_model == "frictionless_spring")
              return c3::multibody::ContactModel::kFrictionlessSpring;
            return c3::multibody::ContactModel::kUnknown;
          },
          [](LCSFactoryOptions& self, c3::multibody::ContactModel val) {
            // Convert enum to the string the C++ struct expects
            switch (val) {
              case c3::multibody::ContactModel::kStewartAndTrinkle:
                self.contact_model = "stewart_and_trinkle";
                break;
              case c3::multibody::ContactModel::kAnitescu:
                self.contact_model = "anitescu";
                break;
              case c3::multibody::ContactModel::kFrictionlessSpring:
                self.contact_model = "frictionless_spring";
                break;
              default:
                self.contact_model = "unknown";
                break;
            }
          })
      .def_property(
          "num_contacts",
          [](const LCSFactoryOptions& self) {
            return self.num_contacts.has_value()
                       ? py::cast(self.num_contacts.value())
                       : py::none();
          },
          [](LCSFactoryOptions& self, py::object val) {
            if (val.is_none()) {
              self.num_contacts.reset();
            } else {
              self.num_contacts = py::cast<int>(val);
            }
          })
      .def_property(
          "spring_stiffness",
          [](const LCSFactoryOptions& self) {
            return self.spring_stiffness.has_value()
                       ? py::cast(self.spring_stiffness.value())
                       : py::none();
          },
          [](LCSFactoryOptions& self, py::object val) {
            if (val.is_none()) {
              self.spring_stiffness.reset();
            } else {
              self.spring_stiffness = py::cast<double>(val);
            }
          })
      .def_property(
          "num_friction_directions",
          [](const LCSFactoryOptions& self) {
            return self.num_friction_directions.has_value()
                       ? py::cast(self.num_friction_directions.value())
                       : py::none();
          },
          [](LCSFactoryOptions& self, py::object val) {
            if (val.is_none()) {
              self.num_friction_directions.reset();
            } else {
              self.num_friction_directions = py::cast<int>(val);
            }
          })
      .def_property(
          "mu",
          [](const LCSFactoryOptions& self) {
            return self.mu.has_value() ? py::cast(self.mu.value()) : py::none();
          },
          [](LCSFactoryOptions& self, py::object val) {
            if (val.is_none()) {
              self.mu.reset();
            } else {
              self.mu = py::cast<double>(val);
            }
          })
      .def_readwrite("num_friction_directions_per_contact",
                     &LCSFactoryOptions::num_friction_directions_per_contact)
      .def_readwrite("mu_per_contact", &LCSFactoryOptions::mu_per_contact)
      .def_readwrite("planar_normal_direction_per_contact",
                     &LCSFactoryOptions::planar_normal_direction_per_contact)
      .def_readwrite("planar_normal_direction",
                     &LCSFactoryOptions::planar_normal_direction)
      .def_readwrite("contact_pair_configs",
                     &LCSFactoryOptions::contact_pair_configs);

  py::class_<c3::multibody::LCSFactory>(m, "LCSFactory")
      .def(py::init<const drake::multibody::MultibodyPlant<double>&,
                    drake::systems::Context<double>&,
                    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>&,
                    drake::systems::Context<drake::AutoDiffXd>&,
                    const std::vector<
                        drake::SortedPair<drake::geometry::GeometryId>>&,
                    const c3::LCSFactoryOptions&>(),
           py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
           py::arg("context_ad"), py::arg("contact_geoms"), py::arg("options"))
      .def(py::init<const drake::multibody::MultibodyPlant<double>&,
                    drake::systems::Context<double>&,
                    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>&,
                    drake::systems::Context<drake::AutoDiffXd>&,
                    c3::LCSFactoryOptions&>(),
           py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
           py::arg("context_ad"), py::arg("options"))
      .def("GenerateLCS", &c3::multibody::LCSFactory::GenerateLCS)
      .def("GetContactDescriptions",
           &c3::multibody::LCSFactory::GetContactDescriptions)
      .def("UpdateStateAndInput",
           &c3::multibody::LCSFactory::UpdateStateAndInput, py::arg("state"),
           py::arg("input"))
      .def_static("LinearizePlantToLCS",
                  &c3::multibody::LCSFactory::LinearizePlantToLCS,
                  py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
                  py::arg("context_ad"), py::arg("contact_geoms"),
                  py::arg("options"), py::arg("state"), py::arg("input"))
      .def_static("GetNClosestContactPairs",
                  &c3::multibody::LCSFactory::GetNClosestContactPairs,
                  py::arg("plant"), py::arg("context"),
                  py::arg("contact_pairs"), py::arg("N"))
      .def_static("FixSomeModes", &c3::multibody::LCSFactory::FixSomeModes,
                  py::arg("other"), py::arg("active_lambda_inds"),
                  py::arg("inactive_lambda_inds"))
      // Overload the function GetNumContactVariables
      .def_static(
          "GetNumContactVariables",
          [](const c3::LCSFactoryOptions& options) {
            return c3::multibody::LCSFactory::GetNumContactVariables(options,
                                                                     nullptr);
          },
          py::arg("options"))
      .def_static(
          "GetNumContactVariables",
          [](const c3::LCSFactoryOptions& options,
             const drake::multibody::MultibodyPlant<double>* plant) {
            return c3::multibody::LCSFactory::GetNumContactVariables(options,
                                                                     plant);
          },
          py::arg("options"), py::arg("plant"))
      .def_static("GetNumContactVariables",
                  py::overload_cast<c3::multibody::ContactModel, int, int>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("contact_model"), py::arg("num_contacts"),
                  py::arg("num_friction_directions"))
      .def_static(
          "GetNumContactVariables",
          py::overload_cast<c3::multibody::ContactModel, int, std::vector<int>>(
              &c3::multibody::LCSFactory::GetNumContactVariables),
          py::arg("contact_model"), py::arg("num_contacts"),
          py::arg("num_friction_directions_per_contact"));

  m.def("LoadLCSFactoryOptions", &LoadLCSFactoryOptions);
}
}  // namespace pyc3
}  // namespace multibody
}  // namespace c3
