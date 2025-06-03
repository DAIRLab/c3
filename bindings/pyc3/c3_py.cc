#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_options.h"
#include "core/c3_qp.h"
#include "core/improved_c3.h"
#include "core/lcs.h"

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
      const Eigen::MatrixXd &U, const Eigen::VectorXd &delta_c,
      const Eigen::MatrixXd &E, const Eigen::MatrixXd &F,
      const Eigen::MatrixXd &H, const Eigen::VectorXd &c,
      const int admm_iteration, const int &warm_start_index) override {
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
      .def(py::init<const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &>(),
           py::arg("Q"), py::arg("R"), py::arg("G"), py::arg("U"))
      .def_property(
          "Q", [](C3::CostMatrices const &self) { return self.Q; },
          [](C3::CostMatrices &self, const std::vector<Eigen::MatrixXd> &val) {
            self.Q = val;
          })
      .def_property(
          "R", [](C3::CostMatrices const &self) { return self.R; },
          [](C3::CostMatrices &self, const std::vector<Eigen::MatrixXd> &val) {
            self.R = val;
          })
      .def_property(
          "G", [](C3::CostMatrices const &self) { return self.G; },
          [](C3::CostMatrices &self, const std::vector<Eigen::MatrixXd> &val) {
            self.G = val;
          })
      .def_property(
          "U", [](C3::CostMatrices const &self) { return self.U; },
          [](C3::CostMatrices &self, const std::vector<Eigen::MatrixXd> &val) {
            self.U = val;
          });

  py::class_<C3, PyC3>(m, "C3")
      .def(py::init<const LCS &, const C3::CostMatrices &,
                    const std::vector<Eigen::VectorXd> &, const C3Options &>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))

      .def("Solve",
           [](C3 &self, const Eigen::VectorXd &x0) {
             py::gil_scoped_release
                 release; // Still need this for nested OpenMP calls
             return self.Solve(x0);
           })
      .def("UpdateLCS", &C3::UpdateLCS, py::arg("lcs"))
      .def("UpdateTarget", &C3::UpdateTarget, py::arg("x_des"))
      .def("AddLinearConstraint",
           static_cast<void (C3::*)(
               const Eigen::MatrixXd &, const Eigen::VectorXd &,
               const Eigen::VectorXd &, int)>(&C3::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"))
      .def("AddLinearConstraint",
           static_cast<void (C3::*)(const Eigen::RowVectorXd &, double, double,
                                    int)>(&C3::AddLinearConstraint),
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
      .def(py::init<const LCS &, const C3::CostMatrices &,
                    const std::vector<Eigen::VectorXd> &, const C3Options &>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))
      .def("GetWarmStartDelta", &C3MIQP::GetWarmStartDelta)
      .def("GetWarmStartBinary", &C3MIQP::GetWarmStartBinary);

  py::class_<C3QP, C3>(m, "C3QP")
      .def(py::init<const LCS &, const C3::CostMatrices &,
                    const std::vector<Eigen::VectorXd> &, const C3Options &>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))
      .def("GetWarmStartDelta", &C3QP::GetWarmStartDelta)
      .def("GetWarmStartBinary", &C3QP::GetWarmStartBinary);

  py::class_<LCS>(m, "LCS")
      .def(py::init<const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::VectorXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::VectorXd> &, double>(),
           py::arg("A"), py::arg("B"), py::arg("D"), py::arg("d"), py::arg("E"),
           py::arg("F"), py::arg("H"), py::arg("c"), py::arg("dt"))

      .def(py::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                    const Eigen::MatrixXd &, const Eigen::VectorXd &,
                    const Eigen::MatrixXd &, const Eigen::MatrixXd &,
                    const Eigen::MatrixXd &, const Eigen::VectorXd &,
                    const int &, double>(),
           py::arg("A"), py::arg("B"), py::arg("D"), py::arg("d"), py::arg("E"),
           py::arg("F"), py::arg("H"), py::arg("c"), py::arg("N"),
           py::arg("dt"))
      .def(py::init<const LCS &>(), py::arg("other"))
      .def("Simulate", &LCS::Simulate, py::arg("x_init"), py::arg("u"),
           "Simulate the system for one step")
      .def("A", &LCS::A, py::return_value_policy::copy)
      .def("B", &LCS::B, py::return_value_policy::copy)
      .def("D", &LCS::D, py::return_value_policy::copy)
      .def("d", &LCS::d, py::return_value_policy::copy)
      .def("E", &LCS::E, py::return_value_policy::copy)
      .def("F", &LCS::F, py::return_value_policy::copy)
      .def("H", &LCS::H, py::return_value_policy::copy)
      .def("c", &LCS::c, py::return_value_policy::copy)
      .def("dt", &LCS::dt, py::return_value_policy::copy)
      .def("N", &LCS::N, py::return_value_policy::copy)
      .def("num_states", &LCS::num_states, py::return_value_policy::copy)
      .def("num_inputs", &LCS::num_inputs, py::return_value_policy::copy)
      .def("num_lambdas", &LCS::num_lambdas, py::return_value_policy::copy);

  py::class_<C3Options> cls(m, "C3Options");
  cls.def(py::init<>());
  //  drake::pydrake::DefAttributesUsingSerialize(&cls);
  cls.def_property(
         "Q", [](C3Options const &self) { return self.Q; },
         [](C3Options &self, const Eigen::MatrixXd &val) { self.Q = val; })
      .def_property(
          "R", [](C3Options const &self) { return self.R; },
          [](C3Options &self, const Eigen::MatrixXd &val) { self.R = val; })
      .def_property(
          "G", [](C3Options const &self) { return self.G; },
          [](C3Options &self, const Eigen::MatrixXd &val) { self.G = val; })
      .def_property(
          "U", [](C3Options const &self) { return self.U; },
          [](C3Options &self, const Eigen::MatrixXd &val) { self.U = val; })
      .def_property(
          "g_vector", [](C3Options const &self) { return self.g_vector; },
          [](C3Options &self, const std::vector<double> &val) {
            self.g_vector = val;
          })
      .def_property(
          "u_vector", [](C3Options const &self) { return self.u_vector; },
          [](C3Options &self, const std::vector<double> &val) {
            self.u_vector = val;
          })
      .def_property(
          "admm_iter", [](C3Options const &self) { return self.admm_iter; },
          [](C3Options &self, const int &val) { self.admm_iter = val; })
      .def_property(
          "rho_scale", [](C3Options const &self) { return self.rho_scale; },
          [](C3Options &self, const double &val) { self.rho_scale = val; })
      .def_property(
          "num_threads", [](C3Options const &self) { return self.num_threads; },
          [](C3Options &self, const int &val) { self.num_threads = val; })
      .def_property(
          "delta_option",
          [](C3Options const &self) { return self.delta_option; },
          [](C3Options &self, const int &val) { self.delta_option = val; });

  m.def("LoadC3Options", &LoadC3Options);

  py::class_<ImprovedC3::CostMatrices>(m, "ImprovedC3CostMatrices")
      .def(py::init<const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &,
                    const std::vector<Eigen::MatrixXd> &>(),
           py::arg("Q"), py::arg("R"), py::arg("G"), py::arg("U"))
      .def_property(
          "Q", [](ImprovedC3::CostMatrices const &self) { return self.Q; },
          [](ImprovedC3::CostMatrices &self,
             const std::vector<Eigen::MatrixXd> &val) { self.Q = val; })
      .def_property(
          "R", [](ImprovedC3::CostMatrices const &self) { return self.R; },
          [](ImprovedC3::CostMatrices &self,
             const std::vector<Eigen::MatrixXd> &val) { self.R = val; })
      .def_property(
          "G", [](ImprovedC3::CostMatrices const &self) { return self.G; },
          [](ImprovedC3::CostMatrices &self,
             const std::vector<Eigen::MatrixXd> &val) { self.G = val; })
      .def_property(
          "U", [](ImprovedC3::CostMatrices const &self) { return self.U; },
          [](ImprovedC3::CostMatrices &self,
             const std::vector<Eigen::MatrixXd> &val) { self.U = val; });
  py::class_<ImprovedC3>(m, "ImprovedC3")
      .def(py::init<const LCS &, const ImprovedC3::CostMatrices &,
                    const std::vector<Eigen::VectorXd> &, const C3Options &>(),
           py::arg("LCS"), py::arg("costs"), py::arg("x_desired"),
           py::arg("options"))
      .def("Solve",
           [](ImprovedC3 &self, const Eigen::VectorXd &x0) {
             py::gil_scoped_release
                 release; // Still need this for nested OpenMP calls
             return self.Solve(x0);
           })
      .def("UpdateLCS", &ImprovedC3::UpdateLCS, py::arg("lcs"))
      .def("UpdateTarget", &ImprovedC3::UpdateTarget, py::arg("x_des"))
      .def("AddLinearConstraint",
           static_cast<void (ImprovedC3::*)(
               const Eigen::MatrixXd &, const Eigen::VectorXd &,
               const Eigen::VectorXd &, int)>(&ImprovedC3::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"))
      .def("AddLinearConstraint",
           static_cast<void (ImprovedC3::*)(const Eigen::RowVectorXd &, double,
                                            double, int)>(
               &ImprovedC3::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"))
      .def("RemoveConstraints", &ImprovedC3::RemoveConstraints)
      .def("SetOsqpSolverOptions", &ImprovedC3::SetOsqpSolverOptions,
           py::arg("options"))
      .def("GetFullSolution", &ImprovedC3::GetFullSolution)
      .def("GetStateSolution", &ImprovedC3::GetStateSolution)
      .def("GetForceSolution", &ImprovedC3::GetForceSolution)
      .def("GetSDFSolution", &ImprovedC3::GetSDFSolution)
      .def("GetInputSolution", &ImprovedC3::GetInputSolution)
      .def("GetDualDeltaSolution", &ImprovedC3::GetDualDeltaSolution)
      .def("GetDualWSolution", &ImprovedC3::GetDualWSolution);
}
} // namespace pyc3
} // namespace c3