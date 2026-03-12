"""Smoke test for all pyc3 bindings."""

import copy
import os
import tempfile

import numpy as np
import c3


def make_lcs(n_x=4, n_u=2, n_lambda=2, N=3, dt=0.01):
    A = np.eye(n_x)
    B = np.zeros((n_x, n_u))
    D = np.zeros((n_x, n_lambda))
    d = np.zeros(n_x)
    E = np.zeros((n_lambda, n_x))
    F = np.eye(n_lambda)
    H = np.zeros((n_lambda, n_u))
    c_vec = np.ones(n_lambda)
    return c3.LCS(A, B, D, d, E, F, H, c_vec, N, dt)


def test_lcs_simulate_config():
    cfg = c3.LCSSimulateConfig()
    cfg.regularized = False
    cfg.piv_tol = 1e-8
    cfg.zero_tol = 1e-10
    cfg.min_exp = -4
    cfg.step_exp = 1
    cfg.max_exp = 4
    print("LCSSimulateConfig OK")


def test_lcs():
    n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
    lcs = make_lcs(n_x, n_u, n_lambda, N, dt)

    assert lcs.num_states() == n_x
    assert lcs.num_inputs() == n_u
    assert lcs.num_lambdas() == n_lambda
    assert lcs.N() == N
    assert lcs.dt() == dt

    # Accessors
    _ = lcs.A(), lcs.B(), lcs.D(), lcs.d()
    _ = lcs.E(), lcs.F(), lcs.H(), lcs.c()

    # Setters (expect lists of N matrices/vectors)
    lcs.set_A([np.eye(n_x)] * N)
    lcs.set_B([np.zeros((n_x, n_u))] * N)
    lcs.set_D([np.zeros((n_x, n_lambda))] * N)
    lcs.set_d([np.zeros(n_x)] * N)
    lcs.set_E([np.zeros((n_lambda, n_x))] * N)
    lcs.set_F([np.eye(n_lambda)] * N)
    lcs.set_H([np.zeros((n_lambda, n_u))] * N)
    lcs.set_c([np.ones(n_lambda)] * N)

    # Simulate
    x0 = np.zeros(n_x)
    u = np.zeros(n_u)
    x_next = lcs.Simulate(x0, u)
    assert x_next.shape == (n_x,)

    x_next_cfg = lcs.Simulate(x0, u, c3.LCSSimulateConfig())
    assert x_next_cfg.shape == (n_x,)

    # List constructor
    A_list = [np.eye(n_x)] * N
    B_list = [np.zeros((n_x, n_u))] * N
    D_list = [np.zeros((n_x, n_lambda))] * N
    d_list = [np.zeros(n_x)] * N
    E_list = [np.zeros((n_lambda, n_x))] * N
    F_list = [np.eye(n_lambda)] * N
    H_list = [np.zeros((n_lambda, n_u))] * N
    c_list = [np.ones(n_lambda)] * N
    lcs2 = c3.LCS(A_list, B_list, D_list, d_list, E_list, F_list, H_list, c_list, dt)
    assert lcs2.num_states() == n_x

    # Copy constructor
    lcs3 = c3.LCS(lcs)
    assert lcs3.num_states() == n_x

    # __copy__ / __deepcopy__
    lcs4 = copy.copy(lcs)
    lcs5 = copy.deepcopy(lcs)
    assert lcs4.num_states() == n_x
    assert lcs5.num_states() == n_x

    # Placeholder
    lcs_ph = c3.LCS.CreatePlaceholderLCS(n_x, n_u, n_lambda, N, dt)
    assert lcs_ph.num_states() == n_x

    print("LCS OK")


def test_cost_matrices():
    n_x, n_u, n_lambda, N = 4, 2, 2, 3
    Q = [np.eye(n_x)] * (N + 1)
    R = [np.eye(n_u)] * N
    G = [np.eye(n_lambda)] * N
    U = [np.eye(n_u)] * N

    cm = c3.CostMatrices(Q, R, G, U)
    assert len(cm.Q) == N + 1
    assert len(cm.R) == N
    assert len(cm.G) == N
    assert len(cm.U) == N

    cm.Q = Q
    cm.R = R
    cm.G = G
    cm.U = U

    cm_default = c3.CostMatrices()
    print("CostMatrices OK")


def test_c3_options():
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
    print("C3Options OK")


def test_constraint_variable():
    assert c3.ConstraintVariable.STATE is not None
    assert c3.ConstraintVariable.INPUT is not None
    assert c3.ConstraintVariable.FORCE is not None
    print("ConstraintVariable OK")


def make_options(n_x=4, n_u=2, n_lambda=2, N=3):
    opts = c3.C3Options()
    opts.Q = np.eye(n_x)
    opts.R = np.eye(n_u)
    opts.G = np.eye(n_x + n_u + n_lambda)
    opts.U = np.eye(n_x + n_u + n_lambda)
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


def test_c3qp():
    n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
    lcs = make_lcs(n_x, n_u, n_lambda, N, dt)
    opts = make_options(n_x, n_u, n_lambda, N)
    costs = c3.C3.CreateCostMatricesFromC3Options(opts, N)

    x_des = [np.zeros(n_x)] * (N + 1)
    solver = c3.C3QP(lcs, costs, x_des, opts)

    x0 = np.zeros(n_x)
    solver.Solve(x0)

    _ = solver.GetFullSolution()
    _ = solver.GetStateSolution()
    _ = solver.GetForceSolution()
    _ = solver.GetInputSolution()
    _ = solver.GetDualDeltaSolution()
    _ = solver.GetDualWSolution()
    _ = solver.GetCostMatrices()
    _ = solver.GetTargetCost()
    _ = solver.GetDynamicConstraints()
    _ = solver.GetLinearConstraints()

    solver.UpdateLCS(lcs)
    solver.UpdateTarget(x_des)
    solver.UpdateCostMatrices(costs)

    solver.AddLinearConstraint(
        np.eye(n_x), -np.ones(n_x), np.ones(n_x), c3.ConstraintVariable.STATE
    )
    solver.RemoveConstraints()

    solver.AddLinearConstraint(
        np.eye(n_u), -np.ones(n_u), np.ones(n_u), c3.ConstraintVariable.INPUT
    )
    solver.RemoveConstraints()

    solver.AddLinearConstraint(
        np.eye(n_lambda), -np.ones(n_lambda), np.ones(n_lambda), c3.ConstraintVariable.FORCE
    )
    solver.RemoveConstraints()

    print("C3QP OK")


def test_c3miqp():
    n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
    lcs = make_lcs(n_x, n_u, n_lambda, N, dt)
    opts = make_options(n_x, n_u, n_lambda, N)
    costs = c3.C3.CreateCostMatricesFromC3Options(opts, N)
    x_des = [np.zeros(n_x)] * (N + 1)
    solver = c3.C3MIQP(lcs, costs, x_des, opts)

    x0 = np.zeros(n_x)
    solver.Solve(x0)

    _ = solver.GetStateSolution()
    _ = solver.GetForceSolution()
    _ = solver.GetInputSolution()

    print("C3MIQP OK")


def test_c3plus():
    n_x, n_u, n_lambda, N, dt = 4, 2, 2, 3, 0.01
    lcs = make_lcs(n_x, n_u, n_lambda, N, dt)
    opts = make_options(n_x, n_u, n_lambda, N)
    costs = c3.C3.CreateCostMatricesFromC3Options(opts, N)
    x_des = [np.zeros(n_x)] * (N + 1)
    solver = c3.C3Plus(lcs, costs, x_des, opts)

    x0 = np.zeros(n_x)
    solver.Solve(x0)

    _ = solver.GetStateSolution()
    _ = solver.GetForceSolution()
    _ = solver.GetInputSolution()

    print("C3Plus OK")


def test_create_cost_matrices_from_options():
    n_x, n_u, n_lambda, N = 4, 2, 2, 3
    opts = make_options(n_x, n_u, n_lambda, N)
    costs = c3.C3.CreateCostMatricesFromC3Options(opts, N)
    assert len(costs.Q) == N + 1
    assert len(costs.R) == N
    print("CreateCostMatricesFromC3Options OK")


def test_load_c3_options():
    """Test LoadC3Options by writing a minimal YAML and loading it."""
    yaml_content = """\
warm_start: false
scale_lcs: false
end_on_qp_step: false
num_threads: 1
delta_option: 0
M: 100.0
admm_iter: 5
gamma: 0.1
rho_scale: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        fname = f.name
    try:
        opts = c3.LoadC3Options(fname)
        assert opts.admm_iter == 5
        assert opts.M == 100.0
        assert abs(opts.gamma - 0.1) < 1e-9
    finally:
        os.unlink(fname)
    print("LoadC3Options OK")


if __name__ == "__main__":
    test_lcs_simulate_config()
    test_lcs()
    test_cost_matrices()
    test_c3_options()
    test_constraint_variable()
    test_c3qp()
    test_c3miqp()
    test_c3plus()
    test_create_cost_matrices_from_options()
    test_load_c3_options()
    print("\nAll smoke tests passed.")
