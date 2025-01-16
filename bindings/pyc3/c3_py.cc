#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "core/c3_miqp.h"
#include "core/lcs.h"

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

        // Bind the two constructors
        // the py::arg arguments aren't strictly necessary, but they allow the python
        // code to use the C3MQP(A: xxx, B: xxx) style


        lcs.def(py::init<const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<MatrixXd> &, const vector<VectorXd> &,
                const vector<MatrixXd> &, const vector<MatrixXd> &,
                const vector<MatrixXd> &, const vector<VectorXd> &, double>(),
                arg("A"), arg("B"), arg("D"), arg("d"), arg("E"), arg("F"),
                arg("H"), arg("c"), arg("dt")
        )
        .def("Simulate", &LCS::Simulate);

    }

}    // namespace pyc3
} // namespace c3


