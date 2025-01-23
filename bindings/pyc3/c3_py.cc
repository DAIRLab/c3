#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/lcs.h"
#include "core/c3.h"
#include "core/c3_qp.h"
#include "core/c3_miqp.h"
#include "core/c3_options.h"

namespace py = pybind11;

namespace c3 {
namespace pyc3 {


/*!
 * Trampoline class to support binding C3's vitrual functions
 * https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 */
class PyC3 : public C3 {
 public:
  /* Inherit the constructors */
  using C3::C3;

  Eigen::VectorXd SolveSingleProjection(
      const Eigen::MatrixXd& U, const Eigen::VectorXd& delta_c,
      const Eigen::MatrixXd& E, const Eigen::MatrixXd& F,
      const Eigen::MatrixXd& H, const Eigen::VectorXd& c,
      const int admm_iteration, const int& warm_start_index) override {
    PYBIND11_OVERRIDE_PURE(Eigen::VectorXd,       /* Return type */
                           C3,                    /* Parent class */
                           SolveSingleProjection, /* Name of function in C++
                                                     (must match Python name) */
                           U, delta_c, E, F, H, c, admm_iteration,
                           warm_start_index /* Argument(s) */
    );
  };
};

PYBIND11_MODULE(c3, m) {
  py::class_<C3::CostMatrices>(m, "CostMatrices")
      .def(py::init<const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&>(),
           py::arg("Q"), py::arg("R"), py::arg("G"), py::arg("U"))
      .def_property(
          "Q", [](C3::CostMatrices const& self) { return self.Q; },
          [](C3::CostMatrices& self, const std::vector<Eigen::MatrixXd>& val) {
            self.Q = val;
          })
      .def_property(
          "R", [](C3::CostMatrices const& self) { return self.R; },
          [](C3::CostMatrices& self, const std::vector<Eigen::MatrixXd>& val) {
            self.R = val;
          })
      .def_property(
          "G", [](C3::CostMatrices const& self) { return self.G; },
          [](C3::CostMatrices& self, const std::vector<Eigen::MatrixXd>& val) {
            self.G = val;
          })
      .def_property(
          "U", [](C3::CostMatrices const& self) { return self.U; },
          [](C3::CostMatrices& self, const std::vector<Eigen::MatrixXd>& val) {
            self.U = val;
          });

  py::class_<C3, PyC3>(m, "C3")
      .def(py::init<const LCS&, const C3::CostMatrices&,
                    const std::vector<Eigen::VectorXd>&, const C3Options&>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))

      .def("Solve",
           [](C3& self, const Eigen::VectorXd& x0) {
             py::gil_scoped_release
                 release;  // Still need this for nested OpenMP calls
             return self.Solve(x0);
           })
      .def("UpdateLCS", &C3::UpdateLCS, py::arg("lcs"))
      .def("UpdateTarget", &C3::UpdateTarget, py::arg("x_des"))
      .def("AddLinearConstraint",
           static_cast<void (C3::*)(
               const Eigen::MatrixXd&,
               const Eigen::VectorXd&,
               const Eigen::VectorXd&, int)>(&C3::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"))
      .def("AddLinearConstraint",
      static_cast<void (C3::*)(
          const Eigen::RowVectorXd&, double, double, int)>(
              &C3::AddLinearConstraint),
          py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
          py::arg("constraint"))
      .def("RemoveConstraints", &C3::RemoveConstraints)
      .def("SetOsqpSolverOptions", &C3::SetOsqpSolverOptions,
           py::arg("options"))
      .def("GetFullSolution", &C3::GetFullSolution)
      .def("GetStateSolution", &C3::GetStateSolution)
      .def("GetForceSolution", &C3::GetForceSolution)
      .def("GetInputSolution", &C3::GetInputSolution);


  py::class_<C3MIQP, C3>(m, "C3MIQP")
      .def(py::init<const LCS&, const C3::CostMatrices&,
                    const std::vector<Eigen::VectorXd>&, const C3Options&>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))
      .def("GetWarmStartDelta", &C3MIQP::GetWarmStartDelta)
      .def("GetWarmStartBinary", &C3MIQP::GetWarmStartBinary);

  py::class_<C3QP, C3>(m, "C3QP")
      .def(py::init<const LCS&, const C3::CostMatrices&,
                    const std::vector<Eigen::VectorXd>&, const C3Options&>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))
      .def("GetWarmStartDelta", &C3QP::GetWarmStartDelta)
      .def("GetWarmStartBinary", &C3QP::GetWarmStartBinary);

  py::class_<LCS>(m, "LCS")
      .def(py::init<const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::VectorXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::MatrixXd>&,
                    const std::vector<Eigen::VectorXd>&, double>(),
           py::arg("A"), py::arg("B"), py::arg("D"), py::arg("d"), py::arg("E"),
           py::arg("F"), py::arg("H"), py::arg("c"), py::arg("dt"))

      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::VectorXd&,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::VectorXd&, const int&,
                    double>(),
           py::arg("A"), py::arg("B"), py::arg("D"), py::arg("d"), py::arg("E"),
           py::arg("F"), py::arg("H"), py::arg("c"), py::arg("N"),
           py::arg("dt"))
      .def(py::init<const LCS&>(), py::arg("other"))

      .def("simulate", &LCS::Simulate, py::arg("x_init"), py::arg("input"),
           "Simulate the system for one step");

  py::class_<C3Options> cls(m, "C3Options");
  cls.def(py::init<>());
  //  drake::pydrake::DefAttributesUsingSerialize(&cls);
  cls.def_property(
         "Q", [](C3Options const& self) { return self.Q; },
         [](C3Options& self, const Eigen::MatrixXd& val) { self.Q = val; })
      .def_property(
          "R", [](C3Options const& self) { return self.R; },
          [](C3Options& self, const Eigen::MatrixXd& val) { self.R = val; })
      .def_property(
          "G", [](C3Options const& self) { return self.G; },
          [](C3Options& self, const Eigen::MatrixXd& val) { self.G = val; })
      .def_property(
          "U", [](C3Options const& self) { return self.U; },
          [](C3Options& self, const Eigen::MatrixXd& val) { self.U = val; })
      .def_property(
          "g_vector", [](C3Options const& self) { return self.g_vector; },
          [](C3Options& self, const std::vector<double>& val) {
            self.g_vector = val;
          })
      .def_property(
          "u_vector", [](C3Options const& self) { return self.u_vector; },
          [](C3Options& self, const std::vector<double>& val) {
            self.u_vector = val;
          });

  m.def("LoadC3Options", &LoadC3Options);
}
}  // namespace pydairlib
}  // namespace dairlib