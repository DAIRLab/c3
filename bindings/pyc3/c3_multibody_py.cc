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
      .def_static("FixSomeModes", &c3::multibody::LCSFactory::FixSomeModes,
                  py::arg("other"), py::arg("active_lambda_inds"),
                  py::arg("inactive_lambda_inds"))
      // Overload the function GetNumContactVariables
      .def_static("GetNumContactVariables",
                  py::overload_cast<c3::multibody::ContactModel, int, int>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("contact_model"), py::arg("num_contacts"),
                  py::arg("num_friction_directions"))
      .def_static("GetNumContactVariables",
                  py::overload_cast<const c3::LCSFactoryOptions>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("options"));

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
                     &ContactPairConfig::planar_normal_direction);

  py::class_<LCSFactoryOptions>(m, "LCSFactoryOptions")
      .def(py::init<>())
      .def_readwrite("dt", &LCSFactoryOptions::dt)
      .def_readwrite("N", &LCSFactoryOptions::N)
      .def_readwrite("contact_model", &LCSFactoryOptions::contact_model)
      .def_readwrite("num_friction_directions",
                     &LCSFactoryOptions::num_friction_directions)
      .def_readwrite("num_friction_directions_per_contact",
                     &LCSFactoryOptions::num_friction_directions_per_contact)
      .def_readwrite("num_contacts", &LCSFactoryOptions::num_contacts)
      .def_readwrite("spring_stiffness", &LCSFactoryOptions::spring_stiffness)
      .def_readwrite("mu", &LCSFactoryOptions::mu)
      .def_readwrite("planar_normal_direction",
                     &LCSFactoryOptions::planar_normal_direction)
      .def_readwrite("planar_normal_direction_per_contact",
                     &LCSFactoryOptions::planar_normal_direction_per_contact)
      .def_readwrite("contact_pair_configs",
                     &LCSFactoryOptions::contact_pair_configs);

  m.def("LoadLCSFactoryOptions", &LoadLCSFactoryOptions);
}
}  // namespace pyc3
}  // namespace multibody
}  // namespace c3
