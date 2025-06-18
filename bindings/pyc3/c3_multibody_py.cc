#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "core/lcs.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/lcs_factory.h"
#include "multibody/multibody_utils.h"

namespace py = pybind11;

PYBIND11_MODULE(c3_multibody_py, m) {
  m.doc() = "C3 Multibody Utilities";

  // GeomGeomCollider Class
  py::class_<c3::multibody::GeomGeomCollider<double>>(m, "GeomGeomCollider")
      .def(py::init<const drake::multibody::MultibodyPlant<double>&,
                    const drake::SortedPair<drake::geometry::GeometryId>>())
      .def("Eval", &c3::multibody::GeomGeomCollider<double>::Eval,
           py::arg("context"),
           py::arg("wrt") = drake::multibody::JacobianWrtVariable::kV)
      .def("EvalPolytope",
           &c3::multibody::GeomGeomCollider<double>::EvalPolytope,
           py::arg("context"), py::arg("num_friction_directions"),
           py::arg("wrt") = drake::multibody::JacobianWrtVariable::kV)
      .def("EvalPlanar", &c3::multibody::GeomGeomCollider<double>::EvalPlanar,
           py::arg("context"), py::arg("planar_normal"),
           py::arg("wrt") = drake::multibody::JacobianWrtVariable::kV)
      .def("CalcWitnessPoints",
           &c3::multibody::GeomGeomCollider<double>::CalcWitnessPoints,
           py::arg("context"));

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
      .def_static("LinearizePlantToLCS",
                  &c3::multibody::LCSFactory::LinearizePlantToLCS,
                  py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
                  py::arg("context_ad"), py::arg("contact_geoms"),
                  py::arg("options"))
      .def_static("ComputeContactJacobian",
                  &c3::multibody::LCSFactory::ComputeContactJacobian,
                  py::arg("plant"), py::arg("context"),
                  py::arg("contact_geoms"), py::arg("options"))
      .def_static("FixSomeModes", &c3::multibody::LCSFactory::FixSomeModes,
                  py::arg("other"), py::arg("active_lambda_inds"),
                  py::arg("inactive_lambda_inds"))
      .def_static("GetNumContactVariables",
                  py::overload_cast<c3::multibody::ContactModel, int, int>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("contact_model"), py::arg("num_contacts"),
                  py::arg("num_friction_directions"))
      .def_static("GetNumContactVariables",
                  py::overload_cast<const c3::LCSOptions>(
                      &c3::multibody::LCSFactory::GetNumContactVariables),
                  py::arg("options"));
}