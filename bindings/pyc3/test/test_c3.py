"""Smoke tests for c3 core bindings.

These tests are primarily smoke tests designed to verify that the Python
bindings are working correctly and can be called without errors. They are
not comprehensive functional tests of the underlying C++ optimization algorithms.
The tests use simple synthetic problems and a realistic cartpole example to
ensure basic binding functionality rather than extensively validating solver
performance, mathematical correctness, or advanced optimization features.
"""

import copy
import os
import unittest
from scipy import linalg

import numpy as np
import c3


def _data_path(relative):
    """Resolve a runfiles data path relative to the workspace root."""
    return os.path.join(os.environ.get("TEST_SRCDIR", "."), "_main", relative)


def make_cartpole_lcs(N=5, dt=0.01):
    g = 9.81
    mp = 0.411
    mc = 0.978
    len_p = 0.6
    len_com = 0.4267
    d1 = 0.35
    d2 = -0.35
    ks = 100
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g * mp / mc, 0, 0],
            [0, g * (mc + mp) / (len_com * mc), 0, 0],
        ]
    )
    A = np.eye(4) + dt * A
    B = dt * np.array([[0], [0], [1 / mc], [1 / (len_com * mc)]])
    D = dt * np.array(
        [
            [0, 0],
            [0, 0],
            [(-1 / mc) + (len_p / (mc * len_com)), (1 / mc) - (len_p / (mc * len_com))],
            [
                (-1 / (mc * len_com))
                + (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
                -(
                    (-1 / (mc * len_com))
                    + (len_p * (mc + mp)) / (mc * mp * len_com * len_com)
                ),
            ],
        ]
    )
    E = np.array([[-1, len_p, 0, 0], [1, -len_p, 0, 0]])
    F = (1.0 / ks) * np.eye(2)
    c_vec = np.array([d1, -d2])
    d = np.zeros(4)
    H = np.zeros((2, 1))
    return c3.LCS(A, B, D, d, E, F, H, c_vec, N, dt)


def make_cartpole_options_and_costs(lcs, N=5, c3plus=False):
    n_x = lcs.num_states()   # 4
    n_u = lcs.num_inputs()   # 1
    n_lambda = lcs.num_lambdas()  # 2

    # State cost: penalize cart pos, pole angle, velocities
    Q_diag = np.diag([10.0, 2.0, 1.0, 1.0])
    # Terminal cost via DARE
    Q_f = linalg.solve_discrete_are(lcs.A()[0], lcs.B()[0], Q_diag, np.eye(n_u))
    R_mat = np.eye(n_u)

    n_z = n_x + n_lambda + n_u
    if c3plus:
        n_z += n_lambda  # extra eta terms for C3+
    G_mat = np.diag(
        [0.1] * n_x + [0.1] * n_lambda + [0.0] * n_u
        + ([1.0] * n_lambda if c3plus else [])
    )
    U_mat = np.diag(
        [1000.0] * n_x + [1.0] * n_lambda + [0.0] * n_u
        + ([10000.0] * n_lambda if c3plus else [])
    )

    Q_list = [Q_diag] * N + [Q_f]
    R_list = [R_mat] * N
    G_list = [G_mat] * N
    U_list = [U_mat] * N

    costs = c3.CostMatrices(Q_list, R_list, G_list, U_list)

    opts = c3.C3Options()
    opts.Q = Q_diag
    opts.R = R_mat
    opts.G = G_mat
    opts.U = U_mat
    opts.g_vector = [0.1] * n_lambda + [0.0] * n_u
    opts.u_vector = [1.0] * n_lambda + [0.0] * n_u
    opts.warm_start = False
    opts.scale_lcs = False
    opts.end_on_qp_step = True
    opts.num_threads = 5
    opts.admm_iter = 10
    opts.M = 1000.0
    opts.gamma = 1.0
    opts.rho_scale = 2.0
    opts.delta_option = 0

    return opts, costs


# keep the simple synthetic helpers for non-solver tests
def make_lcs(n_x=4, n_u=2, n_lambda=2, N=3, dt=0.01):
    return c3.LCS(
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


def make_options(n_x=4, n_u=2, n_lambda=2, is_c3plus=False):
    opts = c3.C3Options()
    opts.Q = np.eye(n_x)
    opts.R = np.eye(n_u)
    n_z = n_x + n_u + n_lambda + (n_lambda if is_c3plus else 0)
    opts.G = np.ones((n_z, n_z))
    opts.U = np.ones((n_z, n_z))
    opts.g_vector = [1.0] * n_lambda
    opts.u_vector = [1.0] * n_u
    opts.warm_start = False
    opts.scale_lcs = False
    opts.end_on_qp_step = False
    opts.num_threads = 1
    opts.admm_iter = 3
    opts.M = 100.0
    opts.gamma = 0.1
    opts.rho_scale = 1.0
    return opts

class TestLCSSimulateConfig(unittest.TestCase):
    def test_fields(self):
        cfg = c3.LCSSimulateConfig()
        cfg.regularized = False
        cfg.piv_tol = 1e-8
        cfg.zero_tol = 1e-10
        cfg.min_exp = -4
        cfg.step_exp = 1
        cfg.max_exp = 4
        self.assertFalse(cfg.regularized)
        self.assertAlmostEqual(cfg.piv_tol, 1e-8)


class TestLCS(unittest.TestCase):
    def setUp(self):
        self.n_x, self.n_u, self.n_lambda, self.N, self.dt = 4, 2, 2, 3, 0.01
        self.lcs = make_lcs(self.n_x, self.n_u, self.n_lambda, self.N, self.dt)

    def test_dimensions(self):
        lcs = self.lcs
        self.assertEqual(lcs.num_states(), self.n_x)
        self.assertEqual(lcs.num_inputs(), self.n_u)
        self.assertEqual(lcs.num_lambdas(), self.n_lambda)
        self.assertEqual(lcs.N(), self.N)
        self.assertAlmostEqual(lcs.dt(), self.dt)

    def test_accessors(self):
        lcs = self.lcs
        self.assertEqual(lcs.A()[0].shape, (self.n_x, self.n_x))
        self.assertEqual(lcs.B()[0].shape, (self.n_x, self.n_u))
        self.assertEqual(lcs.D()[0].shape, (self.n_x, self.n_lambda))
        self.assertEqual(lcs.d()[0].shape, (self.n_x,))
        self.assertEqual(lcs.E()[0].shape, (self.n_lambda, self.n_x))
        self.assertEqual(lcs.F()[0].shape, (self.n_lambda, self.n_lambda))
        self.assertEqual(lcs.H()[0].shape, (self.n_lambda, self.n_u))
        self.assertEqual(lcs.c()[0].shape, (self.n_lambda,))

    def test_setters(self):
        n_x, n_u, n_lambda, N = self.n_x, self.n_u, self.n_lambda, self.N
        lcs = self.lcs
        lcs.set_A([np.eye(n_x)] * N)
        lcs.set_B([np.zeros((n_x, n_u))] * N)
        lcs.set_D([np.zeros((n_x, n_lambda))] * N)
        lcs.set_d([np.zeros(n_x)] * N)
        lcs.set_E([np.zeros((n_lambda, n_x))] * N)
        lcs.set_F([np.eye(n_lambda)] * N)
        lcs.set_H([np.zeros((n_lambda, n_u))] * N)
        lcs.set_c([np.ones(n_lambda)] * N)

    def test_simulate(self):
        x0 = np.zeros(self.n_x)
        u = np.zeros(self.n_u)
        x_next = self.lcs.Simulate(x0, u)
        self.assertEqual(x_next.shape, (self.n_x,))
        x_next2 = self.lcs.Simulate(x0, u, c3.LCSSimulateConfig())
        self.assertEqual(x_next2.shape, (self.n_x,))

    def test_list_constructor(self):
        n_x, n_u, n_lambda, N, dt = self.n_x, self.n_u, self.n_lambda, self.N, self.dt
        lcs2 = c3.LCS(
            [np.eye(n_x)] * N,
            [np.zeros((n_x, n_u))] * N,
            [np.zeros((n_x, n_lambda))] * N,
            [np.zeros(n_x)] * N,
            [np.zeros((n_lambda, n_x))] * N,
            [np.eye(n_lambda)] * N,
            [np.zeros((n_lambda, n_u))] * N,
            [np.ones(n_lambda)] * N,
            dt,
        )
        self.assertEqual(lcs2.num_states(), n_x)

    def test_copy_constructors(self):
        lcs3 = c3.LCS(self.lcs)
        self.assertEqual(lcs3.num_states(), self.n_x)
        lcs4 = copy.copy(self.lcs)
        self.assertEqual(lcs4.num_states(), self.n_x)
        lcs5 = copy.deepcopy(self.lcs)
        self.assertEqual(lcs5.num_states(), self.n_x)

    def test_placeholder(self):
        lcs_ph = c3.LCS.CreatePlaceholderLCS(
            self.n_x, self.n_u, self.n_lambda, self.N, self.dt
        )
        self.assertEqual(lcs_ph.num_states(), self.n_x)


class TestCostMatrices(unittest.TestCase):
    def test_construction_and_properties(self):
        n_x, n_u, n_lambda, N = 4, 2, 2, 3
        Q = [np.eye(n_x)] * (N + 1)
        R = [np.eye(n_u)] * N
        G = [np.eye(n_lambda)] * N
        U = [np.eye(n_u)] * N
        cm = c3.CostMatrices(Q, R, G, U)
        self.assertEqual(len(cm.Q), N + 1)
        self.assertEqual(len(cm.R), N)
        self.assertEqual(len(cm.G), N)
        self.assertEqual(len(cm.U), N)
        cm.Q = Q
        cm.R = R
        cm.G = G
        cm.U = U

    def test_default_construction(self):
        cm = c3.CostMatrices()
        self.assertIsNotNone(cm)


class TestC3Options(unittest.TestCase):
    def test_fields(self):
        opts = c3.C3Options()
        opts.warm_start = False
        opts.scale_lcs = False
        opts.end_on_qp_step = False
        opts.num_threads = 1
        opts.delta_option = 0
        opts.M = 100.0
        opts.admm_iter = 10
        opts.gamma = 0.1
        opts.rho_scale = 1.0
        self.assertFalse(opts.warm_start)
        self.assertEqual(opts.admm_iter, 10)
        self.assertAlmostEqual(opts.gamma, 0.1)

    def test_matrix_properties(self):
        opts = make_options()
        np.testing.assert_array_equal(opts.Q, np.eye(4))
        np.testing.assert_array_equal(opts.R, np.eye(2))


class TestConstraintVariable(unittest.TestCase):
    def test_values(self):
        self.assertIsNotNone(c3.ConstraintVariable.STATE)
        self.assertIsNotNone(c3.ConstraintVariable.INPUT)
        self.assertIsNotNone(c3.ConstraintVariable.FORCE)


class TestC3QP(unittest.TestCase):
    def setUp(self):
        self.N = 5
        self.lcs = make_cartpole_lcs(self.N)
        self.n_x = self.lcs.num_states()
        self.n_u = self.lcs.num_inputs()
        self.n_lambda = self.lcs.num_lambdas()
        self.opts, self.costs = make_cartpole_options_and_costs(self.lcs, self.N)
        self.x_des = [np.zeros(self.n_x)] * (self.N + 1)
        self.solver = c3.C3QP(self.lcs, self.costs, self.x_des, self.opts)

    def test_solve_and_solutions(self):
        self.solver.Solve(np.zeros(self.n_x))
        self.assertIsNotNone(self.solver.GetFullSolution())
        self.assertIsNotNone(self.solver.GetStateSolution())
        self.assertIsNotNone(self.solver.GetForceSolution())
        self.assertIsNotNone(self.solver.GetInputSolution())
        self.assertIsNotNone(self.solver.GetDualDeltaSolution())
        self.assertIsNotNone(self.solver.GetDualWSolution())

    def test_get_cost_matrices(self):
        costs = self.solver.GetCostMatrices()
        self.assertEqual(len(costs.Q), self.N + 1)

    def test_get_target_cost(self):
        self.solver.Solve(np.zeros(self.n_x))
        result = self.solver.GetTargetCost()
        self.assertIsNotNone(result)

    def test_get_dynamic_constraints(self):
        self.solver.Solve(np.zeros(self.n_x))
        result = self.solver.GetDynamicConstraints()
        self.assertIsNotNone(result)

    def test_get_linear_constraints(self):
        result = self.solver.GetLinearConstraints()
        self.assertIsNotNone(result)

    def test_update_methods(self):
        self.solver.UpdateLCS(self.lcs)
        self.solver.UpdateTarget(self.x_des)
        self.solver.UpdateCostMatrices(self.costs)

    def test_add_remove_constraints(self):
        n_x, n_u, n_lambda = self.n_x, self.n_u, self.n_lambda
        self.solver.AddLinearConstraint(
            np.eye(n_x), -np.ones(n_x), np.ones(n_x), c3.ConstraintVariable.STATE
        )
        self.solver.RemoveConstraints()
        self.solver.AddLinearConstraint(
            np.eye(n_u), -np.ones(n_u), np.ones(n_u), c3.ConstraintVariable.INPUT
        )
        self.solver.RemoveConstraints()
        self.solver.AddLinearConstraint(
            np.eye(n_lambda),
            -np.ones(n_lambda),
            np.ones(n_lambda),
            c3.ConstraintVariable.FORCE,
        )
        self.solver.RemoveConstraints()

    def test_create_cost_matrices_from_options(self):
        costs = c3.C3.CreateCostMatricesFromC3Options(self.opts, self.N)
        self.assertEqual(len(costs.Q), self.N + 1)
        self.assertEqual(len(costs.R), self.N)


class TestC3MIQP(unittest.TestCase):
    def test_solve(self):
        N = 5
        lcs = make_cartpole_lcs(N)
        opts, costs = make_cartpole_options_and_costs(lcs, N)
        n_x = lcs.num_states()
        solver = c3.C3MIQP(lcs, costs, [np.zeros(n_x)] * (N + 1), opts)
        solver.Solve(np.zeros(n_x))
        self.assertIsNotNone(solver.GetStateSolution())
        self.assertIsNotNone(solver.GetForceSolution())
        self.assertIsNotNone(solver.GetInputSolution())


class TestC3Plus(unittest.TestCase):
    def test_solve(self):
        N = 5
        lcs = make_cartpole_lcs(N)
        opts, costs = make_cartpole_options_and_costs(lcs, N, c3plus=True)
        n_x = lcs.num_states()
        solver = c3.C3Plus(lcs, costs, [np.zeros(n_x)] * (N + 1), opts)
        solver.Solve(np.zeros(n_x))
        self.assertIsNotNone(solver.GetStateSolution())
        self.assertIsNotNone(solver.GetForceSolution())
        self.assertIsNotNone(solver.GetInputSolution())


class TestLoadC3Options(unittest.TestCase):
    def test_load(self):
        fname = _data_path("core/test/resources/c3_cartpole_options.yaml")
        opts = c3.LoadC3Options(fname)
        self.assertIsNotNone(opts)
        self.assertGreater(opts.admm_iter, 0)
        self.assertGreater(opts.M, 0)


if __name__ == "__main__":
    unittest.main()
