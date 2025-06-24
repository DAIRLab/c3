#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "core/lcs.h"
#include "multibody/lcs_factory_options.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"
#include "multibody/multibody_utils.h"

#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"

#define PYBIND11_DETAILED_ERROR_MESSAGES

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
      .def("GetContactJacobianAndPoints",
           &c3::multibody::LCSFactory::GetContactJacobianAndPoints)
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

  py::class_<LCSFactoryOptions>(m, "LCSFactoryOptions")
      .def(py::init<>())
      .def_readwrite("dt", &LCSFactoryOptions::dt)
      .def_readwrite("N", &LCSFactoryOptions::N)
      .def_readwrite("contact_model", &LCSFactoryOptions::contact_model)
      .def_readwrite("num_friction_directions",
                     &LCSFactoryOptions::num_friction_directions)
      .def_readwrite("num_contacts", &LCSFactoryOptions::num_contacts)
      .def_readwrite("spring_stiffness", &LCSFactoryOptions::spring_stiffness)
      .def_readwrite("mu", &LCSFactoryOptions::mu);

  m.def("LoadLCSFactoryOptions", &LoadLCSFactoryOptions);
}
}  // namespace pyc3
}  // namespace multibody
}  // namespace c3