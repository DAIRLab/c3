#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/lcs.h"
#include "core/traj_eval.h"

namespace py = pybind11;

namespace c3 {
namespace pyc3 {

PYBIND11_MODULE(traj_eval, m) {
  // NOTE: C++ API Flexibility vs. Python Bindings Limitation
  // The C++ TrajectoryEvaluator::ComputeQuadraticTrajectoryCost() uses variadic
  // templates to support arbitrary input combinations (e.g.,
  // ComputeQuadraticTrajectoryCost(x, x_des, Q, u, R)). However, Python's
  // pybind11 does not support variadic template forwarding natively. Therefore,
  // the Python bindings only expose a finite set of predefined overloads. If
  // additional input combinations are needed for Python use cases, they must be
  // explicitly added to this file with corresponding static_cast declarations.
  //
  // IMPORTANT: Binding registration order matters for overload resolution.
  // pybind11 will try to convert a numpy 2D array to std::vector<MatrixXd>
  // (a vector of row vectors) before trying Eigen::MatrixXd. Therefore,
  // single-matrix (Eigen::MatrixXd) overloads MUST be registered before
  // vector-of-matrices (std::vector<Eigen::MatrixXd>) overloads.

  // TrajectoryEvaluator class bindings
  py::class_<traj_eval::TrajectoryEvaluator>(m, "TrajectoryEvaluator")

      // --- ComputeQuadraticTrajectoryCost overloads ---
      // Single-matrix overloads registered FIRST to prevent pybind11 from
      // incorrectly converting a single numpy matrix to vector<MatrixXd>.

      // Binding: (vector<VectorXd>, VectorXd, MatrixXd)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const Eigen::VectorXd& data_des,
             const Eigen::MatrixXd& cost_matrix) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, data_des, cost_matrix);
          },
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrix"),
          "Compute quadratic cost with single desired vector and single cost "
          "matrix.")
      // Binding: (vector<VectorXd>, vector<VectorXd>, MatrixXd)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const std::vector<Eigen::VectorXd>& data_des,
             const Eigen::MatrixXd& cost_matrix) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, data_des, cost_matrix);
          },
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrix"),
          "Compute quadratic cost with single cost matrix.")
      // Binding: (vector<VectorXd>, MatrixXd)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const Eigen::MatrixXd& cost_matrix) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, cost_matrix);
          },
          py::arg("data"), py::arg("cost_matrix"),
          "Compute quadratic cost with single matrix and zero desired.")

      // Vector-of-matrices overloads registered AFTER single-matrix overloads.

      // Binding: (vector<VectorXd>, VectorXd, vector<MatrixXd>)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const Eigen::VectorXd& data_des,
             const std::vector<Eigen::MatrixXd>& cost_matrices) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, data_des, cost_matrices);
          },
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrices"),
          "Compute quadratic cost with single desired vector.")
      // Binding: (vector<VectorXd>, vector<VectorXd>, vector<MatrixXd>)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const std::vector<Eigen::VectorXd>& data_des,
             const std::vector<Eigen::MatrixXd>& cost_matrices) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, data_des, cost_matrices);
          },
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrices"),
          "Compute quadratic cost for full trajectory cost computation.")
      // Binding: (vector<VectorXd>, vector<MatrixXd>)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](const std::vector<Eigen::VectorXd>& data,
             const std::vector<Eigen::MatrixXd>& cost_matrices) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(data, cost_matrices);
          },
          py::arg("data"), py::arg("cost_matrices"),
          "Compute quadratic cost assuming zero desired data.")

      // C3-based overloads: single-matrix overloads first.

      // Binding: (C3*)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](C3* c3) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(c3);
          },
          py::arg("c3"), "Compute cost from C3 optimizer solution.")
      // Binding: (C3*, MatrixXd, MatrixXd)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](C3* c3, const Eigen::MatrixXd& Q,
             const Eigen::MatrixXd& R) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(c3, Q, R);
          },
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single Q and R matrices.")
      // Binding: (C3*, MatrixXd, vector<MatrixXd>)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](C3* c3, const Eigen::MatrixXd& Q,
             const std::vector<Eigen::MatrixXd>& R) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(c3, Q, R);
          },
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single Q matrix.")
      // Binding: (C3*, vector<MatrixXd>, MatrixXd)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](C3* c3, const std::vector<Eigen::MatrixXd>& Q,
             const Eigen::MatrixXd& R) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(c3, Q, R);
          },
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single R matrix.")
      // Binding: (C3*, vector<MatrixXd>, vector<MatrixXd>)
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          [](C3* c3, const std::vector<Eigen::MatrixXd>& Q,
             const std::vector<Eigen::MatrixXd>& R) -> double {
            return traj_eval::TrajectoryEvaluator::
                ComputeQuadraticTrajectoryCost(c3, Q, R);
          },
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with custom cost matrices.")

      // SimulatePDControlWithLCS overloads
      .def_static(
          "SimulatePDControlWithLCS",
          static_cast<std::pair<std::vector<Eigen::VectorXd>,
                                std::vector<Eigen::VectorXd>> (*)(
              const std::vector<Eigen::VectorXd>&,
              const std::vector<Eigen::VectorXd>&, const Eigen::VectorXd&,
              const Eigen::VectorXd&, const LCS&)>(
              &traj_eval::TrajectoryEvaluator::SimulatePDControlWithLCS),
          py::arg("x_plan"), py::arg("u_plan"), py::arg("Kp"), py::arg("Kd"),
          py::arg("lcs"), "Simulate PD control with an LCS")
      .def_static(
          "SimulatePDControlWithLCS",
          static_cast<std::pair<std::vector<Eigen::VectorXd>,
                                std::vector<Eigen::VectorXd>> (*)(
              const std::vector<Eigen::VectorXd>&,
              const std::vector<Eigen::VectorXd>&, const Eigen::VectorXd&,
              const Eigen::VectorXd&, const LCS&, const LCS&)>(
              &traj_eval::TrajectoryEvaluator::SimulatePDControlWithLCS),
          py::arg("x_plan"), py::arg("u_plan"), py::arg("Kp"), py::arg("Kd"),
          py::arg("coarse_lcs"), py::arg("fine_lcs"),
          "Simulate PD control with coarse and fine LCSs")

      // SimulateLCSOverTrajectory overloads
      .def_static(
          "SimulateLCSOverTrajectory",
          static_cast<std::vector<Eigen::VectorXd> (*)(
              const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&,
              const LCS&, const c3::LCSSimulateConfig&)>(
              &traj_eval::TrajectoryEvaluator::SimulateLCSOverTrajectory),
          py::arg("x_init"), py::arg("u_plan"), py::arg("lcs"),
          py::arg("config") = c3::LCSSimulateConfig(),
          "Simulate LCS over input trajectory from initial state")
      .def_static(
          "SimulateLCSOverTrajectory",
          static_cast<std::vector<Eigen::VectorXd> (*)(
              const std::vector<Eigen::VectorXd>&,
              const std::vector<Eigen::VectorXd>&, const LCS&,
              const c3::LCSSimulateConfig&)>(
              &traj_eval::TrajectoryEvaluator::SimulateLCSOverTrajectory),
          py::arg("x_plan"), py::arg("u_plan"), py::arg("lcs"),
          py::arg("config") = c3::LCSSimulateConfig(),
          "Simulate LCS over trajectory using initial state from plan")
      .def_static(
          "SimulateLCSOverTrajectory",
          static_cast<std::vector<Eigen::VectorXd> (*)(
              const Eigen::VectorXd&, const std::vector<Eigen::VectorXd>&,
              const LCS&, const LCS&, const c3::LCSSimulateConfig&)>(
              &traj_eval::TrajectoryEvaluator::SimulateLCSOverTrajectory),
          py::arg("x_init"), py::arg("u_plan"), py::arg("coarse_lcs"),
          py::arg("fine_lcs"), py::arg("config") = c3::LCSSimulateConfig(),
          "Simulate with coarse and fine LCSs from initial state")
      .def_static(
          "SimulateLCSOverTrajectory",
          static_cast<std::vector<Eigen::VectorXd> (*)(
              const std::vector<Eigen::VectorXd>&,
              const std::vector<Eigen::VectorXd>&, const LCS&, const LCS&,
              const c3::LCSSimulateConfig&)>(
              &traj_eval::TrajectoryEvaluator::SimulateLCSOverTrajectory),
          py::arg("x_plan"), py::arg("u_plan"), py::arg("coarse_lcs"),
          py::arg("fine_lcs"), py::arg("config") = c3::LCSSimulateConfig(),
          "Simulate with coarse and fine LCSs using plan initial state")

      // ZeroOrderHold methods
      .def_static("ZeroOrderHoldTrajectories",
                  &traj_eval::TrajectoryEvaluator::ZeroOrderHoldTrajectories,
                  py::arg("x_coarse"), py::arg("u_coarse"),
                  py::arg("upsample_rate"),
                  "Upsample state and input trajectories")
      .def_static("ZeroOrderHoldTrajectory",
                  &traj_eval::TrajectoryEvaluator::ZeroOrderHoldTrajectory,
                  py::arg("coarse_traj"), py::arg("upsample_rate"),
                  "Upsample a single trajectory")

      // Downsample methods
      .def_static("DownsampleTrajectories",
                  &traj_eval::TrajectoryEvaluator::DownsampleTrajectories,
                  py::arg("x_fine"), py::arg("u_fine"),
                  py::arg("downsample_rate"),
                  "Downsample state and input trajectories")
      .def_static("DownsampleTrajectory",
                  &traj_eval::TrajectoryEvaluator::DownsampleTrajectory,
                  py::arg("fine_traj"), py::arg("downsample_rate"),
                  "Downsample a single trajectory")

      // LCS compatibility checking methods
      .def_static(
          "CheckCoarseAndFineLCSCompatibility",
          &traj_eval::TrajectoryEvaluator::CheckCoarseAndFineLCSCompatibility,
          py::arg("coarse_lcs"), py::arg("fine_lcs"),
          "Check LCS time discretization compatibility")
      .def_static("CheckLCSAndTrajectoryCompatibility",
                  static_cast<void (*)(
                      const LCS&, const std::vector<Eigen::VectorXd>&,
                      const std::optional<std::vector<Eigen::VectorXd>>&,
                      const std::optional<std::vector<Eigen::VectorXd>>&)>(
                      &traj_eval::TrajectoryEvaluator::
                          CheckLCSAndTrajectoryCompatibility),
                  py::arg("lcs"), py::arg("x"), py::arg("u") = py::none(),
                  py::arg("lambda") = py::none(),
                  "Check trajectory compatibility with LCS; inputs and lambdas "
                  "are optional");
}
}  // namespace pyc3
}  // namespace c3
