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

  // LCSSimulateConfig struct bindings
  py::class_<LCSSimulateConfig>(m, "LCSSimulateConfig")
      .def(py::init<>())
      .def_readwrite("regularized", &LCSSimulateConfig::regularized)
      .def_readwrite("piv_tol", &LCSSimulateConfig::piv_tol)
      .def_readwrite("zero_tol", &LCSSimulateConfig::zero_tol)
      .def_readwrite("min_exp", &LCSSimulateConfig::min_exp)
      .def_readwrite("step_exp", &LCSSimulateConfig::step_exp)
      .def_readwrite("max_exp", &LCSSimulateConfig::max_exp);

  // TrajectoryEvaluator class bindings
  py::class_<traj_eval::TrajectoryEvaluator>(m, "TrajectoryEvaluator")

      // ComputeQuadraticTrajectoryCost overloads
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(const std::vector<Eigen::VectorXd>&,
                                 const std::vector<Eigen::VectorXd>&,
                                 const std::vector<Eigen::MatrixXd>&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrices"),
          "Compute quadratic cost for full trajectory cost computation")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(const std::vector<Eigen::VectorXd>&,
                                 const std::vector<Eigen::VectorXd>&,
                                 const Eigen::MatrixXd&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("data"), py::arg("data_des"), py::arg("cost_matrix"),
          "Compute quadratic cost with single cost matrix")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(const std::vector<Eigen::VectorXd>&,
                                 const std::vector<Eigen::MatrixXd>&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("data"), py::arg("cost_matrices"),
          "Compute quadratic cost assuming zero desired data")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(const std::vector<Eigen::VectorXd>&,
                                 const Eigen::MatrixXd&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("data"), py::arg("cost_matrix"),
          "Compute quadratic cost with single matrix and zero desired")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(C3*)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("c3"), "Compute cost from C3 optimizer solution")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(C3*, const std::vector<Eigen::MatrixXd>&,
                                 const std::vector<Eigen::MatrixXd>&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with custom cost matrices")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(C3*, const Eigen::MatrixXd&,
                                 const std::vector<Eigen::MatrixXd>&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single Q matrix")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(C3*, const std::vector<Eigen::MatrixXd>&,
                                 const Eigen::MatrixXd&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single R matrix")
      .def_static(
          "ComputeQuadraticTrajectoryCost",
          static_cast<double (*)(C3*, const Eigen::MatrixXd&,
                                 const Eigen::MatrixXd&)>(
              &traj_eval::TrajectoryEvaluator::ComputeQuadraticTrajectoryCost),
          py::arg("c3"), py::arg("Q"), py::arg("R"),
          "Compute cost from C3 with single Q and R matrices")

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
                  py::arg("upsample_rate"),
                  "Downsample state and input trajectories")
      .def_static("DownsampleTrajectory",
                  &traj_eval::TrajectoryEvaluator::DownsampleTrajectory,
                  py::arg("fine_traj"), py::arg("upsample_rate"),
                  "Downsample a single trajectory")

      // LCS compatibility checking methods
      .def_static(
          "CheckCoarseAndFineLCSCompatibility",
          &traj_eval::TrajectoryEvaluator::CheckCoarseAndFineLCSCompatibility,
          py::arg("coarse_lcs"), py::arg("fine_lcs"),
          "Check LCS time discretization compatibility")
      .def_static(
          "CheckLCSAndTrajectoryCompatibility",
          static_cast<void (*)(const LCS&, const std::vector<Eigen::VectorXd>&,
                               const std::vector<Eigen::VectorXd>&,
                               const std::vector<Eigen::VectorXd>&)>(
              &traj_eval::TrajectoryEvaluator::
                  CheckLCSAndTrajectoryCompatibility),
          py::arg("lcs"), py::arg("x"), py::arg("u"), py::arg("lambda"),
          "Check trajectory compatibility with LCS (with lambdas)")
      .def_static(
          "CheckLCSAndTrajectoryCompatibility",
          static_cast<void (*)(const LCS&, const std::vector<Eigen::VectorXd>&,
                               const std::vector<Eigen::VectorXd>&)>(
              &traj_eval::TrajectoryEvaluator::
                  CheckLCSAndTrajectoryCompatibility),
          py::arg("lcs"), py::arg("x"), py::arg("u"),
          "Check trajectory compatibility with LCS (without lambdas)")
      .def_static("CheckLCSAndTrajectoryCompatibility",
                  static_cast<void (*)(const LCS&,
                                       const std::vector<Eigen::VectorXd>&)>(
                      &traj_eval::TrajectoryEvaluator::
                          CheckLCSAndTrajectoryCompatibility),
                  py::arg("lcs"), py::arg("x"),
                  "Check state trajectory compatibility with LCS");
}
}  // namespace pyc3
}  // namespace c3
