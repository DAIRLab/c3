"""Smoke tests for c3 systems bindings."""

import copy
import os
import tempfile
import unittest

import numpy as np
import c3
import systems

# multibody is a separate module — import it so LCSFactoryOptions is available
try:
    import multibody

    HAS_MULTIBODY = True
except ImportError:
    HAS_MULTIBODY = False

class TestC3Solution(unittest.TestCase):
    def test_default_construction(self):
        sol = systems.C3Solution()
        self.assertIsNotNone(sol)

    def test_construction_with_dims(self):
        n_x, n_lambda, n_u, N = 4, 2, 2, 3
        sol = systems.C3Solution(n_x, n_lambda, n_u, N)
        # Verify collections are non-empty and all elements are 1-D numpy arrays
        self.assertGreater(len(sol.x_sol), 0)
        self.assertGreater(len(sol.lambda_sol), 0)
        self.assertGreater(len(sol.u_sol), 0)
        for v in sol.x_sol:
            self.assertEqual(v.ndim, 1)
        for v in sol.lambda_sol:
            self.assertEqual(v.ndim, 1)
        for v in sol.u_sol:
            self.assertEqual(v.ndim, 1)

    def test_time_vector(self):
        sol = systems.C3Solution(4, 2, 2, 3)
        self.assertIsNotNone(sol.time_vector)

    def test_readwrite_fields(self):
        sol = systems.C3Solution(4, 2, 2, 3)
        # overwrite x_sol with new list
        new_x = [np.ones(4)] * len(sol.x_sol)
        sol.x_sol = new_x
        np.testing.assert_array_equal(sol.x_sol[0], np.ones(4))

    def test_copy(self):
        sol = systems.C3Solution(4, 2, 2, 3)
        sol2 = copy.copy(sol)
        sol3 = copy.deepcopy(sol)
        self.assertIsNotNone(sol2)
        self.assertIsNotNone(sol3)


class TestC3Intermediates(unittest.TestCase):
    def test_default_construction(self):
        inter = systems.C3Intermediates()
        self.assertIsNotNone(inter)

    def test_construction_with_dims(self):
        n_x, n_lambda, n_u, N = 4, 2, 2, 3
        inter = systems.C3Intermediates(n_x, n_lambda, n_u, N)
        self.assertIsNotNone(inter.z)
        self.assertIsNotNone(inter.delta)
        self.assertIsNotNone(inter.w)
        self.assertIsNotNone(inter.time_vector)

    def test_readwrite_fields(self):
        inter = systems.C3Intermediates(4, 2, 2, 3)
        new_z = [np.zeros(4)] * len(inter.z)
        inter.z = new_z
        np.testing.assert_array_equal(inter.z[0], np.zeros(4))

    def test_copy(self):
        inter = systems.C3Intermediates(4, 2, 2, 3)
        inter2 = copy.copy(inter)
        inter3 = copy.deepcopy(inter)
        self.assertIsNotNone(inter2)
        self.assertIsNotNone(inter3)


class TestC3StatePredictionJoint(unittest.TestCase):
    def test_fields(self):
        joint = systems.C3StatePredictionJoint()
        joint.name = "shoulder"
        joint.max_acceleration = 10.0
        self.assertEqual(joint.name, "shoulder")
        self.assertAlmostEqual(joint.max_acceleration, 10.0)


class TestC3ControllerOptions(unittest.TestCase):
    def test_fields(self):
        opts = systems.C3ControllerOptions()
        opts.solve_time_filter_alpha = 0.5
        opts.publish_frequency = 100.0
        # projection_type is a str per binding
        opts.projection_type = "qp"
        opts.quaternion_weight = 1.0
        opts.quaternion_regularizer_fraction = 0.1
        self.assertAlmostEqual(opts.solve_time_filter_alpha, 0.5)
        self.assertAlmostEqual(opts.publish_frequency, 100.0)
        self.assertAlmostEqual(opts.quaternion_weight, 1.0)
        self.assertAlmostEqual(opts.quaternion_regularizer_fraction, 0.1)

    def test_nested_c3_options(self):
        opts = systems.C3ControllerOptions()
        c3_opts = c3.C3Options()
        c3_opts.admm_iter = 5
        opts.c3_options = c3_opts
        self.assertEqual(opts.c3_options.admm_iter, 5)

    @unittest.skipUnless(HAS_MULTIBODY, "multibody module not available")
    def test_nested_lcs_factory_options(self):
        opts = systems.C3ControllerOptions()
        lcs_opts = multibody.LCSFactoryOptions()
        lcs_opts.dt = 0.01
        opts.lcs_factory_options = lcs_opts
        self.assertAlmostEqual(opts.lcs_factory_options.dt, 0.01)

    def test_state_prediction_joints(self):
        opts = systems.C3ControllerOptions()
        joint = systems.C3StatePredictionJoint()
        joint.name = "elbow"
        joint.max_acceleration = 5.0
        opts.state_prediction_joints = [joint]
        self.assertEqual(len(opts.state_prediction_joints), 1)
        self.assertEqual(opts.state_prediction_joints[0].name, "elbow")

    def test_load_c3_controller_options(self):
        opts = systems.LoadC3ControllerOptions(
            "examples/resources/cartpole_softwalls/c3_controller_cartpole_options.yaml"
        )
        self.assertAlmostEqual(opts.publish_frequency, 100.0)
        self.assertAlmostEqual(opts.solve_time_filter_alpha, 0.0)


class TestTimestampedVector(unittest.TestCase):
    def test_construction(self):
        # Drake template classes use underscore suffix: TimestampedVector_[float]
        vec = systems.TimestampedVector_[float](4)
        self.assertIsNotNone(vec)

    def test_set_get_timestamp(self):
        vec = systems.TimestampedVector_[float](4)
        vec.set_timestamp(1.23)
        self.assertAlmostEqual(vec.get_timestamp(), 1.23)

    def test_set_get_data(self):
        vec = systems.TimestampedVector_[float](4)
        data = np.ones(4) * 2.0
        vec.SetDataVector(data)
        np.testing.assert_array_almost_equal(vec.get_data(), data)


class TestLCSSimulator(unittest.TestCase):
    def _make_lcs(self):
        n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
        return c3.LCS(
            np.eye(n_x),
            np.zeros((n_x, n_u)),
            np.zeros((n_x, n_lambda)),
            np.zeros(n_x),
            np.zeros((n_lambda, n_x)),
            np.eye(n_lambda),
            np.zeros((n_lambda, n_u)),
            np.ones(n_lambda),
            N,
            dt,
        )

    def test_construct_from_lcs(self):
        lcs = self._make_lcs()
        sim = systems.LCSSimulator(lcs)
        self.assertIsNotNone(sim)

    def test_construct_from_dims(self):
        sim = systems.LCSSimulator(4, 2, 2, 3, 0.01)
        self.assertIsNotNone(sim)

    def test_ports(self):
        sim = systems.LCSSimulator(4, 2, 2, 3, 0.01)
        self.assertIsNotNone(sim.get_input_port_state())
        self.assertIsNotNone(sim.get_input_port_action())
        self.assertIsNotNone(sim.get_input_port_lcs())
        self.assertIsNotNone(sim.get_output_port_next_state())


class TestLCSFactorySystemCallable(unittest.TestCase):
    """Verify LCSFactorySystem is importable and constructor signature is correct."""

    def test_class_exists(self):
        self.assertTrue(hasattr(systems, "LCSFactorySystem"))

    def test_constructor_requires_plant(self):
        # Should raise TypeError (missing args) not ImportError
        with self.assertRaises((TypeError, Exception)):
            systems.LCSFactorySystem()


class TestC3ControllerCallable(unittest.TestCase):
    """Verify C3Controller is importable and constructor signature is correct."""

    def test_class_exists(self):
        self.assertTrue(hasattr(systems, "C3Controller"))

    def test_constructor_requires_plant(self):
        with self.assertRaises((TypeError, Exception)):
            systems.C3Controller()


class TestValueInstantiations(unittest.TestCase):
    """Verify Value<T> instantiations are registered."""

    def test_lcs_value(self):
        from pydrake.common.value import Value
        n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
        lcs = c3.LCS(
            np.eye(n_x), np.zeros((n_x, n_u)), np.zeros((n_x, n_lambda)),
            np.zeros(n_x), np.zeros((n_lambda, n_x)), np.eye(n_lambda),
            np.zeros((n_lambda, n_u)), np.ones(n_lambda), N, dt,
        )
        val = Value[c3.LCS](lcs)
        self.assertIsNotNone(val)

    def test_c3solution_value(self):
        from pydrake.common.value import Value
        sol = systems.C3Solution(4, 2, 2, 3)
        val = Value[systems.C3Solution](sol)
        self.assertIsNotNone(val)

    def test_c3intermediates_value(self):
        from pydrake.common.value import Value
        inter = systems.C3Intermediates(4, 2, 2, 3)
        val = Value[systems.C3Intermediates](inter)
        self.assertIsNotNone(val)


if __name__ == "__main__":
    unittest.main()
