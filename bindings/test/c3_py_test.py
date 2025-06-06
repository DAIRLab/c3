import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyc3 import LCS, C3MIQP, C3ControllerOptions, CostMatrices, LoadC3ControllerOptions



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

    return LCS(A, B, D, d, E, F, H, c, N, dt,)


def make_cartpole_costs(lcs:LCS, options: C3ControllerOptions) -> CostMatrices:

    R = [options.R for _ in range(options.N)]
    Q = [options.Q for _ in range(options.N)]
    Q.append(linalg.solve_discrete_are(lcs.A()[0], lcs.B()[0], options.Q, options.R))
    G = [options.G for _ in range(options.N)]
    U = [options.U for _ in range(options.N)]

    return CostMatrices(Q, R, G, U)


def main():
    options = LoadC3ControllerOptions(
        "core/test/res/c3_cartpole_options.yaml"
    )
    cartpole = make_cartpole_with_soft_walls_dynamics(options.N)
    costs = make_cartpole_costs(cartpole, options)

    n = cartpole.num_states()

    x0 = np.zeros((n, 1))
    xd = [x0 for _ in range(options.N + 1)]

    opt = C3MIQP(cartpole, costs, xd, options)
    
    x0[0] = 0.01
    x0[2] = 0.03

    system_iter = 500

    x = np.zeros((n, system_iter+1))

    x[:, 0] = x0.ravel()

    for i in tqdm(range(system_iter)):
        opt.Solve(x[:, i])
        u_opt = opt.GetInputSolution()[0]
        prediction = cartpole.Simulate(x[:, i], u_opt)
        x[:, i+1] = prediction

    dt = cartpole.dt()
    time_x = np.arange(0, system_iter * dt + dt, dt)

    plt.plot(time_x, x.T)
    plt.show()


if __name__ == "__main__":
    main()
