import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

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
    Ginit[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = np.eye(m)
    Ginit[n : n + m, n : n + m] = np.eye(m)
    G = [Ginit for _ in range(N)]

    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.eye(m)
    U[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = 10000 * np.eye(m)
    U = [U for _ in range(N)]

    return ImprovedC3CostMatrices(Q, R, G, U)


def animate_cartpole(x, dt, len_p, len_com):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.25, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw soft walls
    ax.axvline(x=-0.35, color="blue", linestyle="-", alpha=0.5)
    ax.axvline(x=0.35, color="blue", linestyle="-", alpha=0.5)

    # Create lines for cart, pole, and center of mass
    (cart_line,) = ax.plot([], [], "b-", lw=2)
    (pole_line,) = ax.plot([], [], "r-", lw=2)
    (com_marker,) = ax.plot([], [], "go", markersize=8)  # green circle for COM
    time_text = fig.text(
        0.02, 0.95, "", ha="center", fontsize=12, transform=fig.transFigure
    )
    cart_width = 0.2
    cart_height = 0.1

    def init():
        cart_line.set_data([], [])
        pole_line.set_data([], [])
        com_marker.set_data([], [])
        time_text.set_text("")
        return cart_line, pole_line, com_marker, time_text

    def animate(i):
        # Get cart position and pole angle
        cart_pos = x[0, i]
        pole_angle = x[1, i]

        # Calculate cart corners
        cart_x = [
            cart_pos - cart_width / 2,
            cart_pos + cart_width / 2,
            cart_pos + cart_width / 2,
            cart_pos - cart_width / 2,
            cart_pos - cart_width / 2,
        ]
        cart_y = [
            -cart_height / 2,
            -cart_height / 2,
            cart_height / 2,
            cart_height / 2,
            -cart_height / 2,
        ]

        # Calculate pole endpoints
        pole_x = [cart_pos, cart_pos + len_p * np.sin(pole_angle)]
        pole_y = [0, len_p * np.cos(pole_angle)]

        # Calculate center of mass position
        com_x = cart_pos + len_com * np.sin(pole_angle)
        com_y = len_com * np.cos(pole_angle)

        # Update time text
        current_time = i * dt
        time_text.set_text(f"Time: {current_time:.2f}s")

        cart_line.set_data(cart_x, cart_y)
        pole_line.set_data(pole_x, pole_y)
        com_marker.set_data([com_x], [com_y])
        return cart_line, pole_line, com_marker, time_text

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=x.shape[1], interval=1000 / 60, blit=True
    )  # 60 FPS

    # Save animation
    writer = animation.PillowWriter(fps=60)
    anim.save("/home/hienbui/git/c3/cartpole_animation.gif", writer=writer)

    return anim


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

    system_iter = 1000

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    delta_sol = []

    for i in range(system_iter):
        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        delta_sol.append(opt.GetDualDeltaSolution())
        u_opt = opt.GetInputSolution()[0]
        prediction = cartpole.Simulate(x[:, i], u_opt)
        x[:, i + 1] = prediction

    delta_sol = np.array(delta_sol)

    dt = cartpole.dt()

    # Create animation if necessary
    # len_p = 0.6  # pole length
    # len_com = 0.4267  # center of mass length
    # anim = animate_cartpole(x, dt, len_p, len_com)

    time_x = np.arange(0, system_iter * dt + dt, dt)

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    ax[0].plot(time_x, x.T)
    ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("State")
    ax[0].set_title("Improved C3 Controller")

    ax[1].plot(time_x[:-1], delta_sol[:, 0, 4], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 7], label=r"$\gamma_1$")
    ax[1].legend()
    ax[1].set_ylabel(r"Left Wall")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(time_x[:-1], delta_sol[:, 0, 5], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 8], label=r"$\gamma_2$")
    ax[2].legend()
    ax[2].set_ylabel(r"Right Wall")
    ax[2].set_xlabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    main()
