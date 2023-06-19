#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "src/c3_miqp.h"
#include "src/lcs.h"

namespace py = pybind11;

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using py::arg;

namespace c3 {
    namespace pyc3 {


        PYBIND11_MODULE(c3, m
        ) {

        py::class_ <LCS> lcs(m, "LCS");
        py::class_ <C3MIQP> c3miqp(m, "C3MIQP");

        // Bind the two constructors
        // the py::arg arguments aren't strictly necessary, but they allow the python
        // code to use the C3MQP(A: xxx, B: xxx) style


        lcs.def(py::init<const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<MatrixXd> &, const vector<VectorXd> &,
                const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<MatrixXd> &, const vector<VectorXd> &>(),
                arg("A"), arg("B"), arg("D"), arg("d"), arg("E"), arg("F"),
                arg("H"), arg("c")

        )
        .def(py::init<const MatrixXd &, const MatrixXd &, const MatrixXd &,
                const MatrixXd &, const MatrixXd &, const MatrixXd &,
                const MatrixXd &, const VectorXd &, int>(),
                arg("A"), arg("B"), arg("D"), arg("d"), arg("E"), arg("F"),
                arg("H"), arg("c"), arg("N")

        )
        .def("Simulate", &LCS::Simulate);

        c3miqp.
        def(py::init<const LCS &,
                const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<VectorXd> &, const C3Options &,
                const vector<VectorXd> &, const vector<VectorXd> &,
                const vector<VectorXd> &, const vector<VectorXd> &,
                const vector<VectorXd> &, bool>(),
                arg("LCS"), arg("Q"), arg("R"), arg("G"), arg("U"),
                arg("x_desired"), arg("options"),
                arg("warm_start_delta") = vector<VectorXd>(),
                arg("warm_start_binary") = vector<VectorXd>(),
                arg("warm_start_x") = vector<VectorXd>(),
                arg("warm_start_lambda") = vector<VectorXd>(),
                arg("warm_start_u") = vector<VectorXd>(),
                arg("warm_start") = false
        )
        .def_static("MakeTimeInvariantC3MIQP",
                &C3MIQP::MakeTimeInvariantC3MIQP,
                arg("LCS"), arg("Q"), arg("R"), arg("G"), arg("U"),
                arg("x_desired"), arg("Qf"), arg("options"), arg("N"),
                arg("warm_start_delta") = vector<VectorXd>(),
                arg("warm_start_binary") = vector<VectorXd>(),
                arg("warm_start_x") = vector<VectorXd>(),
                arg("warm_start_lambda") = vector<VectorXd>(),
                arg("warm_start_u") = vector<VectorXd>(),
                arg("warm_start") = false
        )
        .def("Solve", &C3MIQP::Solve)
        .def("AddLinearConstraint", &C3MIQP::AddLinearConstraint,
          arg("A"), arg("Lowerbound"), arg("Upperbound"), arg("constraint"))
        .def("RemoveConstraints", &C3MIQP::RemoveConstraints);

        // An example of binding a simple function. pybind will automatically
        // deduce the arguments, but providing the names here for usability

        /*
        c3miqp.def("SolveSingleProjection",
                             &C3MIQP::SolveSingleProjection, arg("U"), arg("delta_c"), arg("E"),
                          arg("F"), arg("H"), arg("c"));
      */

/*

	// For binding Solve, because it has multiple return arguments
	// (via pointer, in C++) pointer, the binding is a bit more complex
  c3miqp.def("Solve",
  		[](C3MIQP* self, VectorXd& x0, vector<VectorXd>* delta,
  			     vector<VectorXd>* w) {
  			VectorXd u = self->Solve(x0, delta, w);
  			// Return a tuple of (u, delta, w), by value
  			//return py::make_tuple(u, *delta, *w);
              return u;

  		}
  		);

*/

        py::class_<C3Options> options(m, "C3Options");
        options.def(py::init<>())
        .def_readwrite("admm_iter", &C3Options::admm_iter)
        .def_readwrite("rho", &C3Options::rho)
        .def_readwrite("OSQP_verbose_flag", &C3Options::OSQP_verbose_flag)
        .def_readwrite("OSQP_maximum_iterations", &C3Options::OSQP_maximum_iterations)
        .def_readwrite("Gurobi_verbose_flag", &C3Options::Gurobi_verbose_flag)
        .def_readwrite("Gurobi_maximum_iterations", &C3Options::Gurobi_maximum_iterations)
        .def_readwrite("rho_scale", &C3Options::rho_scale);


    }

}    // namespace pyc3
} // namespace c3


