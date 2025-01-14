#include "c3_output.h"

using Eigen::VectorXd;
using Eigen::VectorXf;
using std::vector;

namespace c3 {

C3Output::C3Output(C3Solution c3_sol, C3Intermediates c3_intermediates)
    : c3_solution_(c3_sol), c3_intermediates_(c3_intermediates) {}

}  // namespace c3
