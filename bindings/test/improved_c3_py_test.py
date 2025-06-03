import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from pyc3 import (
    LCS,
    C3Options,
    ImprovedC3,
    ImprovedC3CostMatrices,
)


def make_cartpole_with_soft_walls_dynamics(N: int) -> LCS:
    g = 9.81
    mp = 0.411
    mc = 0.978
    len_p = 0.6
    len_com = 0.4267
    d1 = 0.35
    d2 = -0.35
    ks = 100
    dt = 0.01
    A = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, g * mp / mc, 0, 0],
            [0, g * (mc + mp) / (len_com * mc), 0, 0],
        ]
    )
    A = np.eye(A.shape[0]) + dt * A
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
    c = np.array([[d1], [-d2]])
    d = np.zeros((4, 1))
    H = np.zeros((2, 1))

    return LCS(A, B, D, d, E, F, H, c, N, dt)


def make_cartpole_costs(lcs: LCS) -> ImprovedC3CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()

    R = [np.eye(k) for _ in range(N)]
    Q = [
        np.array([[10, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for _ in range(N)
    ]
    Q.append(linalg.solve_discrete_are(lcs.A()[0], lcs.B()[0], Q[0], R[0]))

    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = np.eye(m)
    Ginit[n : n + m, n : n + m] = np.eye(m)
    G = [Ginit for _ in range(N)]

    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.eye(m)
    U[n + m + k : n + m + k + m, n + m + k : n + m + k + m] = 10000 * np.eye(m)
    U = [U for _ in range(N)]

    return ImprovedC3CostMatrices(Q, R, G, U)


def main():
    N = 10
    cartpole = make_cartpole_with_soft_walls_dynamics(N)
    costs = make_cartpole_costs(cartpole)

    n = cartpole.num_states()

    x0 = np.zeros((n, 1))
    xd = [x0 for _ in range(N + 1)]
    options = C3Options()
    options.admm_iter = 10
    options.rho_scale = 2
    options.num_threads = 5
    options.delta_option = 0

    opt = ImprovedC3(cartpole, costs, xd, options)

    x0 = np.array([0, -0.5, 0.5, -0.4])

    system_iter = 600

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    sdf_sol = []

    for i in range(system_iter):
        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        u_opt = opt.GetInputSolution()[0]
        prediction = cartpole.Simulate(x[:, i], u_opt)
        x[:, i + 1] = prediction

    dt = cartpole.dt()
    time_x = np.arange(0, system_iter * dt + dt, dt)

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    plt.plot(time_x, x.T)
    plt.legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title("Improved C3 Controller")
    plt.show()


if __name__ == "__main__":
    main()
