import time
import numpy as np
import matplotlib.pyplot as plt

from pyc3 import (
    LCS,
    C3Options,
    C3MIQP,
    CostMatrices,
)


def is_controllable(A, B):
    n = A.shape[0]
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack(
            (controllability_matrix, np.linalg.matrix_power(A, i) @ B)
        )
    rank = np.linalg.matrix_rank(controllability_matrix)
    return rank == n, rank


def generate_random_lcs(n_x, n_u, n_lambda, N: int, stiffness: float = 1.0) -> LCS:
    A = np.random.uniform(-1, 1, size=(n_x, n_x))
    B = np.random.uniform(-1, 1, size=(n_x, n_u))
    D = np.random.uniform(-1, 1, size=(n_x, n_lambda))
    d = np.random.uniform(-1, 1, size=n_x)

    E = np.random.uniform(-1, 1, size=(n_lambda, n_x))
    H = np.random.uniform(-1, 1, size=(n_lambda, n_u))
    W = np.tril(np.random.uniform(-1, 1, size=(n_lambda, n_lambda)))
    V = np.random.uniform(-1, 1, size=(n_lambda, n_lambda))
    F = W @ W.T + V - V.T + stiffness * np.eye(n_lambda)
    c = 0.01 * np.random.uniform(-1, 1, size=n_lambda)

    assert is_controllable(A, B), "System is not controllable"

    return LCS(A, B, D, d, E, F, H, c, N, 0)


def make_gen_lcs_costs(lcs: LCS) -> CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()

    R = [0.0001 * np.eye(k) for _ in range(N)]
    Q = [50 * np.eye(n) for _ in range(N + 1)]

    Ginit = 0.1 * np.eye(n + m + k)
    Ginit[n : (n + m), n : (n + m)] = 0.02 * np.eye(m)
    G = [Ginit for _ in range(N)]

    U = np.eye(n + m + k)
    U[:n, :n] = np.eye(n)
    U[n : (n + m), n : (n + m)] = 1000 * np.eye(m)
    U[n + m :, n + m :] = np.eye(k)
    U = [U for _ in range(N)]

    return CostMatrices(Q, R, G, U)


def main():
    np.random.seed(0)
    N = 5
    n_x = 6
    n_u = 2
    n_lambda = 30
    stiffness = 4.0
    gen_lcs = generate_random_lcs(n_x, n_u, n_lambda, N, stiffness)
    costs = make_gen_lcs_costs(gen_lcs)

    x0 = np.zeros((n_x, 1))
    xd = [x0 for _ in range(N + 1)]
    options = C3Options()
    options.admm_iter = 5
    options.rho_scale = 3
    options.num_threads = 5

    opt = C3MIQP(gen_lcs, costs, xd, options)

    x0 = np.linspace(-10, 10, n_x)

    system_iter = 50

    x = np.zeros((n_x, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []

    for i in range(system_iter):
        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        u_opt = opt.GetInputSolution()[0]
        prediction = gen_lcs.Simulate(x[:, i], u_opt)
        x[:, i + 1] = prediction

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    # Plot the state trajectory
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax.plot(np.arange(system_iter + 1), x.T)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("States")
    ax.set_title(
        r"C3 Controller on a synthetic LCS with $n_x = 6, n_u = 2, n_{\lambda}=30$"
    )
    ax.grid(True)
    plt.savefig("/home/hienbui/git/c3/c3_random_lcs.png", dpi=600)


if __name__ == "__main__":
    main()
