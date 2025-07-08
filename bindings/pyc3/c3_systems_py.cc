#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/c3.h"
#include "core/lcs.h"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"
#include "systems/lcmt_generators/c3_output_generator.h"
#include "systems/lcmt_generators/c3_trajectory_generator.h"
#include "systems/lcmt_generators/contact_force_generator.h"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"

#include "drake/bindings/pydrake/common/cpp_template_pybind.h"
#include "drake/bindings/pydrake/common/default_scalars_pybind.h"
#include "drake/bindings/pydrake/common/sorted_pair_pybind.h"
#include "drake/bindings/pydrake/common/value_pybind.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/constant_value_source.h"

namespace py = pybind11;

using drake::multibody::MultibodyPlant;
using drake::pydrake::CommonScalarPack;
using drake::pydrake::DefineTemplateClassWithDefault;
using drake::pydrake::GetPyParam;
using drake::systems::BasicVector;
using drake::systems::ConstantValueSource;
using drake::systems::LeafSystem;

namespace c3 {
namespace systems {
namespace pyc3 {
PYBIND11_MODULE(systems, m) {
  py::module::import("pydrake.systems.framework");
  py::class_<C3Controller, LeafSystem<double>>(m, "C3Controller")
      .def(py::init<const MultibodyPlant<double>&, const C3::CostMatrices,
                    C3ControllerOptions>(),
           py::arg("plant"), py::arg("costs"), py::arg("options"))
      .def("get_input_port_target", &C3Controller::get_input_port_target,
           py::return_value_policy::reference)
      .def("get_input_port_lcs_state", &C3Controller::get_input_port_lcs_state,
           py::return_value_policy::reference)
      .def("get_input_port_lcs", &C3Controller::get_input_port_lcs,
           py::return_value_policy::reference)
      .def("get_output_port_c3_solution",
           &C3Controller::get_output_port_c3_solution,
           py::return_value_policy::reference)
      .def("get_output_port_c3_intermediates",
           &C3Controller::get_output_port_c3_intermediates,
           py::return_value_policy::reference)
      .def("UpdateCostMatrices", &C3Controller::UpdateCostMatrices,
           py::arg("cost_matrices"))
      .def("AddLinearConstraint",
           static_cast<void (C3Controller::*)(
               const Eigen::MatrixXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&, enum ConstraintVariable constraint)>(
               &C3Controller::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"))
      .def("AddLinearConstraint",
           static_cast<void (C3Controller::*)(
               const Eigen::RowVectorXd&, double, double,
               enum ConstraintVariable constraint)>(
               &C3Controller::AddLinearConstraint),
           py::arg("A"), py::arg("lower_bound"), py::arg("upper_bound"),
           py::arg("constraint"));

  py::class_<LCSSimulator, LeafSystem<double>>(m, "LCSSimulator")
      .def(py::init<LCS>(), py::arg("lcs"))
      .def(py::init<int, int, int, int, double>(), py::arg("n_x"),
           py::arg("n_u"), py::arg("n_lambda"), py::arg("N"), py::arg("dt"))
      .def("get_input_port_state", &LCSSimulator::get_input_port_state,
           py::return_value_policy::reference)
      .def("get_input_port_action", &LCSSimulator::get_input_port_action,
           py::return_value_policy::reference)
      .def("get_input_port_lcs", &LCSSimulator::get_input_port_lcs,
           py::return_value_policy::reference)
      .def("get_output_port_next_state",
           &LCSSimulator::get_output_port_next_state,
           py::return_value_policy::reference);

  py::class_<LCSFactorySystem, LeafSystem<double>>(m, "LCSFactorySystem")
      .def(
          py::init<
              const MultibodyPlant<double>&, drake::systems::Context<double>&,
              const MultibodyPlant<drake::AutoDiffXd>&,
              drake::systems::Context<drake::AutoDiffXd>&,
              const std::vector<drake::SortedPair<drake::geometry::GeometryId>>,
              LCSFactoryOptions>(),
          py::arg("plant"), py::arg("context"), py::arg("plant_ad"),
          py::arg("context_ad"), py::arg("contact_geoms"), py::arg("options"))
      .def("get_input_port_lcs_state",
           &LCSFactorySystem::get_input_port_lcs_state,
           py::return_value_policy::reference)
      .def("get_input_port_lcs_input",
           &LCSFactorySystem::get_input_port_lcs_input,
           py::return_value_policy::reference)
      .def("get_output_port_lcs", &LCSFactorySystem::get_output_port_lcs,
           py::return_value_policy::reference)
      .def("get_output_port_lcs_contact_description",
           &LCSFactorySystem::get_output_port_lcs_contact_description,
           py::return_value_policy::reference);

  py::class_<C3Output::C3Solution>(m, "C3Solution")
      .def(py::init<>())
      .def(py::init<int, int, int, int>(), py::arg("n_x"), py::arg("n_lambda"),
           py::arg("n_u"), py::arg("N"))
      .def_readwrite("time_vector", &C3Output::C3Solution::time_vector_)
      .def_readwrite("x_sol", &C3Output::C3Solution::x_sol_)
      .def_readwrite("lambda_sol", &C3Output::C3Solution::lambda_sol_)
      .def_readwrite("u_sol", &C3Output::C3Solution::u_sol_)
      .def("__copy__",
           [](const C3Output::C3Solution& self) {
             return C3Output::C3Solution(self);
           })
      .def("__deepcopy__", [](const C3Output::C3Solution& self, py::dict) {
        return C3Output::C3Solution(self);
      });
  ;

  py::class_<C3Output::C3Intermediates>(m, "C3Intermediates")
      .def(py::init<>())
      .def(py::init<int, int, int, int>(), py::arg("n_x"), py::arg("n_lambda"),
           py::arg("n_u"), py::arg("N"))
      .def_readwrite("time_vector", &C3Output::C3Intermediates::time_vector_)
      .def_readwrite("z", &C3Output::C3Intermediates::z_)
      .def_readwrite("delta", &C3Output::C3Intermediates::delta_)
      .def_readwrite("w", &C3Output::C3Intermediates::w_)
      .def("__copy__",
           [](const C3Output::C3Intermediates& self) {
             return C3Output::C3Intermediates(self);
           })
      .def("__deepcopy__", [](const C3Output::C3Intermediates& self, py::dict) {
        return C3Output::C3Intermediates(self);
      });
  ;

  auto bind_common_scalar_types = [m](auto dummy) {
    using T = decltype(dummy);
    DefineTemplateClassWithDefault<TimestampedVector<T>, BasicVector<T>>(
        m, "TimestampedVector", GetPyParam<T>())
        .def(py::init<int>(), py::arg("size"))
        .def("get_timestamp", &TimestampedVector<T>::get_timestamp,
             py::return_value_policy::copy)
        .def("get_data", &TimestampedVector<T>::get_data,
             py::return_value_policy::copy)
        .def("SetDataVector", &TimestampedVector<T>::SetDataVector,
             py::arg("data_vector"))
        .def("set_timestamp", &TimestampedVector<T>::set_timestamp,
             py::arg("timestamp"));
  };
  type_visit(bind_common_scalar_types, CommonScalarPack{});

  drake::pydrake::AddValueInstantiation<c3::LCS>(m);
  drake::pydrake::AddValueInstantiation<C3Output::C3Solution>(m);
  drake::pydrake::AddValueInstantiation<C3Output::C3Intermediates>(m);

  py::class_<C3StatePredictionJoint>(m, "C3StatePredictionJoint")
      .def(py::init<>())
      .def_readwrite("name", &C3StatePredictionJoint::name)
      .def_readwrite("max_acceleration",
                     &C3StatePredictionJoint::max_acceleration);

  py::class_<C3ControllerOptions>(m, "C3ControllerOptions")
      .def(py::init<>())
      .def_readwrite("solve_time_filter_alpha",
                     &C3ControllerOptions::solve_time_filter_alpha)
      .def_readwrite("publish_frequency",
                     &C3ControllerOptions::publish_frequency)
      .def_readwrite("projection_type", &C3ControllerOptions::projection_type)
      .def_readwrite("c3_options", &C3ControllerOptions::c3_options)
      .def_readwrite("lcs_factory_options",
                     &C3ControllerOptions::lcs_factory_options)
      .def_readwrite("state_prediction_joints",
                     &C3ControllerOptions::state_prediction_joints);

  m.def("LoadC3ControllerOptions", &LoadC3ControllerOptions);

  // C3OutputGenerator
  py::class_<lcmt_generators::C3OutputGenerator,
             drake::systems::LeafSystem<double>>(m, "C3OutputGenerator")
      .def(py::init<>())
      .def("get_input_port_c3_solution",
           &lcmt_generators::C3OutputGenerator::get_input_port_c3_solution,
           py::return_value_policy::reference)
      .def("get_input_port_c3_intermediates",
           &lcmt_generators::C3OutputGenerator::get_input_port_c3_intermediates,
           py::return_value_policy::reference)
      .def("get_output_port_c3_output",
           &lcmt_generators::C3OutputGenerator::get_output_port_c3_output,
           py::return_value_policy::reference)
      .def_static("AddLcmPublisherToBuilder",
                  &lcmt_generators::C3OutputGenerator::AddLcmPublisherToBuilder,
                  py::arg("builder"), py::arg("solution_port"),
                  py::arg("intermediates_port"), py::arg("channel"),
                  py::arg("lcm"), py::arg("publish_triggers"),
                  py::arg("publish_period"), py::arg("publish_offset"));

  // C3TrajectoryGenerator
  py::class_<lcmt_generators::TrajectoryDescription::index_range>(
      m, "TrajectoryDescriptionIndexRange")
      .def(py::init<>())
      .def_readwrite(
          "start", &lcmt_generators::TrajectoryDescription::index_range::start)
      .def_readwrite("end",
                     &lcmt_generators::TrajectoryDescription::index_range::end);
  py::class_<lcmt_generators::TrajectoryDescription>(m, "TrajectoryDescription")
      .def(py::init<>())
      .def_readwrite("trajectory_name",
                     &lcmt_generators::TrajectoryDescription::trajectory_name)
      .def_readwrite("variable_type",
                     &lcmt_generators::TrajectoryDescription::variable_type)
      .def_readwrite("indices",
                     &lcmt_generators::TrajectoryDescription::indices);
  py::class_<lcmt_generators::C3TrajectoryGeneratorConfig>(
      m, "C3TrajectoryGeneratorConfig")
      .def(py::init<>())
      .def_readwrite(
          "trajectories",
          &lcmt_generators::C3TrajectoryGeneratorConfig::trajectories);
  py::class_<lcmt_generators::C3TrajectoryGenerator,
             drake::systems::LeafSystem<double>>(m, "C3TrajectoryGenerator")
      .def(py::init<lcmt_generators::C3TrajectoryGeneratorConfig>())
      .def("get_input_port_c3_solution",
           &lcmt_generators::C3TrajectoryGenerator::get_input_port_c3_solution,
           py::return_value_policy::reference)
      .def("get_output_port_lcmt_c3_trajectory",
           &lcmt_generators::C3TrajectoryGenerator::
               get_output_port_lcmt_c3_trajectory,
           py::return_value_policy::reference)
      .def_static(
          "AddLcmPublisherToBuilder",
          &lcmt_generators::C3TrajectoryGenerator::AddLcmPublisherToBuilder,
          py::arg("builder"), py::arg("config"), py::arg("solution_port"),
          py::arg("channel"), py::arg("lcm"), py::arg("publish_triggers"),
          py::arg("publish_period"), py::arg("publish_offset"));

  // ContactForceGenerator
  py::class_<lcmt_generators::ContactForceGenerator,
             drake::systems::LeafSystem<double>>(m, "ContactForceGenerator")
      .def(py::init<>())
      .def("get_input_port_c3_solution",
           &lcmt_generators::ContactForceGenerator::get_input_port_c3_solution,
           py::return_value_policy::reference)
      .def("get_input_port_lcs_contact_info",
           &lcmt_generators::ContactForceGenerator::
               get_input_port_lcs_contact_info,
           py::return_value_policy::reference)
      .def("get_output_port_contact_force",
           &lcmt_generators::ContactForceGenerator::
               get_output_port_contact_force,
           py::return_value_policy::reference)
      .def_static(
          "AddLcmPublisherToBuilder",
          &lcmt_generators::ContactForceGenerator::AddLcmPublisherToBuilder,
          py::arg("builder"), py::arg("solution_port"),
          py::arg("lcs_contact_info_port"), py::arg("channel"), py::arg("lcm"),
          py::arg("publish_triggers"), py::arg("publish_period"),
          py::arg("publish_offset"));
}
}  // namespace pyc3
}  // namespace systems
}  // namespace c3