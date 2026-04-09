"""Smoke tests for c3 multibody bindings.

These tests are primarily smoke tests designed to verify that the Python
bindings are working correctly and can be called without errors. They are
not comprehensive functional tests of the underlying C++ multibody mechanics
or contact modeling algorithms. The tests focus on ensuring that configuration
classes can be loaded, options can be set, and basic utility functions can
be called through the Python interface rather than validating complex
multibody dynamics or contact physics.
"""

import unittest
import multibody
import os


class TestContactModel(unittest.TestCase):
    def test_values(self):
        self.assertIsNotNone(multibody.ContactModel.Unknown)
        self.assertIsNotNone(multibody.ContactModel.StewartAndTrinkle)
        self.assertIsNotNone(multibody.ContactModel.Anitescu)
        self.assertIsNotNone(multibody.ContactModel.FrictionlessSpring)


class TestLCSFactoryOptions(unittest.TestCase):
    def test_fields(self):
        opts = multibody.LCSFactoryOptions()
        opts.dt = 0.01
        opts.N = 3
        opts.num_contacts = 2
        # mu is list[float] per binding
        opts.mu = 0.5
        opts.spring_stiffness = 100.0
        opts.num_friction_directions = 4
        self.assertAlmostEqual(opts.dt, 0.01)
        self.assertEqual(opts.N, 3)
        self.assertEqual(opts.num_contacts, 2)
        self.assertAlmostEqual(opts.mu, 0.5)

    def test_contact_model(self):
        opts = multibody.LCSFactoryOptions()
        opts.contact_model = multibody.ContactModel.StewartAndTrinkle
        self.assertEqual(opts.contact_model, multibody.ContactModel.StewartAndTrinkle)

    def test_contact_pair_configs(self):
        opts = multibody.LCSFactoryOptions()
        cfg = multibody.ContactPairConfig()
        opts.contact_pair_configs = [cfg]
        self.assertEqual(len(opts.contact_pair_configs), 1)


class TestContactPairConfig(unittest.TestCase):
    def test_fields(self):
        cfg = multibody.ContactPairConfig()
        cfg.mu = 0.7
        cfg.num_friction_directions = 2
        cfg.num_active_contact_pairs = 1
        self.assertAlmostEqual(cfg.mu, 0.7)
        self.assertEqual(cfg.num_friction_directions, 2)

    def test_body_fields(self):
        cfg = multibody.ContactPairConfig()
        cfg.body_A = "base"
        cfg.body_B = "link1"
        self.assertEqual(cfg.body_A, "base")
        self.assertEqual(cfg.body_B, "link1")

    def test_geom_indices(self):
        cfg = multibody.ContactPairConfig()
        cfg.body_A_collision_geom_indices = [0, 1]
        cfg.body_B_collision_geom_indices = [2]
        self.assertEqual(cfg.body_A_collision_geom_indices, [0, 1])


class TestLCSFactoryGetNumContactVariables(unittest.TestCase):
    def test_with_contact_model(self):
        n = multibody.LCSFactory.GetNumContactVariables(
            multibody.ContactModel.StewartAndTrinkle, 2, 4
        )
        self.assertGreater(n, 0)

    def test_with_options(self):
        opts = multibody.LCSFactoryOptions()
        opts.num_contacts = 2
        opts.num_friction_directions = 4
        opts.contact_model = multibody.ContactModel.StewartAndTrinkle
        n = multibody.LCSFactory.GetNumContactVariables(opts)
        self.assertGreater(n, 0)


class TestLoadLCSFactoryOptions(unittest.TestCase):
    def test_load(self):
        opts = multibody.LoadLCSFactoryOptions(
            "multibody/test/resources/lcs_factory_pivoting_options.yaml"
        )
        self.assertEqual(opts.N, 10)
        self.assertAlmostEqual(opts.dt, 0.01)
        self.assertEqual(opts.num_contacts, 3)
        self.assertEqual(opts.contact_model, multibody.ContactModel.StewartAndTrinkle)
        self.assertEqual(opts.num_friction_directions, 1)
        self.assertAlmostEqual(opts.mu, 0.1)
        self.assertEqual(len(opts.contact_pair_configs), 3)
        self.assertEqual(opts.contact_pair_configs[0].body_A, "cube")
        self.assertEqual(opts.contact_pair_configs[0].body_B, "left_finger")

    def test_get_num_contact_variables_from_loaded_options(self):
        opts = multibody.LoadLCSFactoryOptions(
            "multibody/test/resources/lcs_factory_pivoting_options.yaml"
        )
        opts.contact_pair_configs = None  # test that GetNumContactVariables doesn't require this field
        n = multibody.LCSFactory.GetNumContactVariables(opts)
        self.assertGreater(n, 0)


class TestLCSFactoryCallable(unittest.TestCase):
    """Verify LCSFactory methods are callable (may raise due to missing plant)."""

    def test_fix_some_modes_callable(self):
        import c3
        import numpy as np

        n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
        lcs = c3.LCS(
            np.eye(n_x),
            np.ones((n_x, n_u)),
            np.ones((n_x, n_lambda)),
            np.ones(n_x),
            np.ones((n_lambda, n_x)),
            np.eye(n_lambda),
            np.ones((n_lambda, n_u)),
            np.ones(n_lambda),
            N,
            dt,
        )
        # FixSomeModes takes sets of ints — just call it
        result = multibody.LCSFactory.FixSomeModes(lcs, {0}, {1})
        self.assertIsNotNone(result)

    def test_get_num_contact_variables_anitescu(self):
        n = multibody.LCSFactory.GetNumContactVariables(
            multibody.ContactModel.Anitescu, 2, 4
        )
        self.assertGreater(n, 0)

    def test_get_num_contact_variables_frictionless(self):
        n = multibody.LCSFactory.GetNumContactVariables(
            multibody.ContactModel.FrictionlessSpring, 2, 0
        )
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
