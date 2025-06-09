#!/usr/bin/env python3
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


def make_fingergait_dynamics(N: int) -> LCS:
    N = 10
    h = 0.1
    g = 9.81
    mu = 1.0
    dt = 0.01
    A = np.array([[1, h, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, h, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, h],
                  [0, 0, 0, 0, 0, 1]])
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [h * h, 0, 0, 0],
                  [h, 0, 0, 0],
                  [0, h * h, 0, 0],
                  [0, h, 0, 0]])
    D = np.array([[0, h * h, -h * h, 0, h * h, -h * h],
                  [0, h, -h, 0, h, -h],
                  [0, -h * h, h * h, 0, 0, 0],
                  [0, -h, h, 0, 0, 0],
                  [0, 0, 0, 0, -h * h, h * h],
                  [0, 0, 0, 0, -h, h]])
    E = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 1, 0, -1, 0, 0],
                  [0, -1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, -1],
                  [0, -1, 0, 0, 0, 1]])
    F = np.array([[0, -1, -1, 0, 0, 0],
                  [1, 2 * h, -2 * h, 0, h, -h],
                  [1, -2 * h, 2 * h, 0, -h, h],
                  [0, 0, 0, 0, -1, -1],
                  [0, h, -h, 1, 2 * h, -2 * h],
                  [0, -h, h, 1, -2 * h, 2 * h]])
    c = np.array([[0], [-h * g], [h * g], [0], [-h * g], [h * g]])
    d = np.array([[-g * h * h], [-g * h], [0], [0], [0], [0]])
    H = np.array([[0, 0, mu, 0], 
                  [-h, 0, 0, 0],
                  [h, 0, 0, 0],
                  [0, 0, 0, mu],
                  [0, -h, 0, 0],
                  [0, h, 0, 0]])
    # G = np.eye(n+m+k)
    return LCS(A, B, D, d, E, F, H, c, N, dt)


def make_fingergait_costs(lcs: LCS) -> ImprovedC3CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()
    print(f"Number of states: {n}, Number of lambdas: {m}, Number of inputs: {k}")

    R = [np.eye(k) for _ in range(N)]
    Q_init = np.array([[6000, 0, 0, 0, 0, 0],
                    [0, 10, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0],
                    [0, 0, 0, 0, 10, 0],
                    [0, 0, 0, 0, 0, 10]])
    Q = [Q_init for _ in range(N + 1)]
    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = np.eye(m)
    Ginit[n : n + m, n : n + m] = np.eye(m)
    G = [Ginit for _ in range(N)]

    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.eye(m)
    U[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = 1000 * np.eye(m)
    U = [U for _ in range(N)]

    return ImprovedC3CostMatrices(Q, R, G, U)


def animate_fingergait(x, dt, len_rod, width_rod):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.25, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    (rod_line,) = ax.plot([], [], "b-", lw=2)
    (limit_line,) = ax.plot([], [], "y-", lw=2)
    (com_marker,) = ax.plot([], [], "go", markersize=8)  # green circle for COM
    (gripper_marker1_left,) = ax.plot([], [], "ro", markersize=6)  # red circle for gripper
    (gripper_marker1_right,) = ax.plot([], [], "ro", markersize=6)  # red circle for gripper
    (gripper_marker2_left,) = ax.plot([], [], "ro", markersize=6)  # red circle for gripper
    (gripper_marker2_right,) = ax.plot([], [], "ro", markersize=6)  # red circle for gripper
    time_text = fig.text(
        0.02, 0.95, "", ha="center", fontsize=12, transform=fig.transFigure
    )

    def init():
        rod_line.set_data([], [])
        limit_line.set_data([], [])
        com_marker.set_data([], [])
        gripper_marker1_left.set_data([], [])
        gripper_marker1_right.set_data([], [])
        gripper_marker2_left.set_data([], [])
        gripper_marker2_right.set_data([], [])
        time_text.set_text("")
        return rod_line, limit_line, com_marker, gripper_marker1_left, gripper_marker1_right, gripper_marker2_left, gripper_marker2_right, time_text

    def animate(i):
        # Get cart position and pole angle
        object_pos = x[0, i]
        g1_pos = x[2, i]
        g2_pos = x[4, i]

        # Calculate cart corners
        rod_x = [
            -width_rod / 2,
            width_rod / 2,
            width_rod / 2,
            -width_rod / 2,
            -width_rod / 2,
        ]
        rod_y = [
            object_pos - len_rod / 2,
            object_pos - len_rod / 2,
            object_pos + len_rod / 2,
            object_pos + len_rod / 2,
            object_pos - len_rod / 2,
        ]

        # Calculate pole endpoints
        gripper1_right_x = width_rod / 2
        gripper1_right_y = g1_pos
        gripper1_left_x = -width_rod / 2
        gripper1_left_y = g1_pos

        gripper2_right_x = width_rod / 2
        gripper2_right_y = g2_pos
        gripper2_left_x = -width_rod / 2
        gripper2_left_y = g2_pos
        # pole_x = [cart_pos, cart_pos - len_p * np.sin(pole_angle)]
        # pole_y = [0, len_p * np.cos(pole_angle)]

        # Calculate center of mass position
        com_x = 0
        com_y = object_pos

        # Update time text
        current_time = i * dt
        time_text.set_text(f"Time: {current_time:.2f}s")

        rod_line.set_data(rod_x, rod_y)
        # pole_line.set_data(pole_x, pole_y)
        com_marker.set_data([com_x], [com_y])
        gripper_marker1_left.set_data([gripper1_left_x], [gripper1_left_y])
        gripper_marker1_right.set_data([gripper1_right_x], [gripper1_right_y])
        gripper_marker2_left.set_data([gripper2_left_x], [gripper2_left_y])
        gripper_marker2_right.set_data([gripper2_right_x], [gripper2_right_y])
        return rod_line, com_marker, gripper_marker1_left, gripper_marker1_right, gripper_marker2_left, gripper_marker2_right, time_text

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=x.shape[1], interval=1000 / 60, blit=True
    )  # 60 FPS

    # Save animation
    writer = animation.PillowWriter(fps=60)
    anim.save("/home/yufeiyang/Documents/c3/fingerGait_animation.gif", writer=writer)

    return anim


def main():
    N = 10
    finger = make_fingergait_dynamics(N)
    costs = make_fingergait_costs(finger)

    n = finger.num_states()
    Xd0 = np.array([[0], [0], [2], [0], [4], [0]])
    xd = [Xd0 for _ in range(N + 1)]
    options = C3Options()
    options.admm_iter = 30
    options.rho_scale = 2.5
    options.num_threads = 10
    options.delta_option = 0

    opt = ImprovedC3(finger, costs, xd, options)

    x0 = np.array([[-8], [0], [3], [0], [4], [0]])

    system_iter = 1000

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    sdf_sol = []
    delta_sol = []
    z_sol = []
    x_ = []
    u_sol = []
    a1 = np.array([0, 0, 1, 0, 0, 0])
    a2 = np.array([0, 0, 0, 0, 1, 0])

    u1 = np.array([0, 0, 1, 0])
    u2 = np.array([0, 0, 0, 1])
    debug_info = []
    debug_qp = []
    opt.AddLinearConstraint(a1, 1, 3, 1)
    opt.AddLinearConstraint(a2, 3, 5, 1)
    opt.AddLinearConstraint(u1, 0, 10000, 2)
    opt.AddLinearConstraint(u2, 0, 10000, 2)
    for i in range(system_iter):
        # whe i = 0; get result from SolveQP
        start_time = time.perf_counter()

        opt.Solve(x[:, i])
        if i == 0:
            debug_info = opt.GetDebugInfo()
            debug_qp = opt.GetQPInfo()
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        delta_sol.append(opt.GetDualDeltaSolution())
        u_opt = opt.GetInputSolution()[0]
        z_sol.append(opt.GetFullSolution())
        prediction = finger.Simulate(x[:, i], u_opt)
        u_sol.append(u_opt)
        x[:, i + 1] = prediction
        x_.append(opt.GetStateSolution())


    sdf_sol = np.array(sdf_sol)
    delta_sol = np.array(delta_sol)
    print(delta_sol.shape)

    dt = finger.dt()

    debug_info = np.array(debug_info)
    debug_qp = np.array(debug_qp)
    print(debug_info.shape)
    # # save debug info to file
    # with open('/home/yufeiyang/Documents/c3/debug_output/debug_info.txt', 'a') as f:
    #     f.write('\n')
    #     np.savetxt(f, debug_info[1], fmt='%s')

    z_sol = np.array(z_sol)
    print(z_sol.shape)
    # Save the results to a file
    np.save('/home/yufeiyang/Documents/c3/debug_output/debug.npy', debug_info)
    np.save('/home/yufeiyang/Documents/c3/debug_output/debug_qp.npy', debug_qp)
    np.save('/home/yufeiyang/Documents/c3/debug_output/z_sol.npy', z_sol)
    np.save('/home/yufeiyang/Documents/c3/debug_output/delta_sol.npy', delta_sol)
    np.save('/home/yufeiyang/Documents/c3/debug_output/x_.npy', x_)

    # Create animation if necessary
    len_p = 0.6  # rod length
    len_com = 0.2  # rod width
    # anim = animate_fingergait(x, dt, len_p, len_com)

    time_x = np.arange(0, system_iter * dt + dt, dt)

    u_sol = np.array(u_sol)
    print(u_sol.shape)

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

    ax[0].plot(time_x, x.T)
    ax[0].legend(["Object position", "Object velocity", "Gripper 1 position", "Gripper 1 velocity",
                 "Gripper 2 position", "Gripper 2 velocity"])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("State")
    ax[0].set_title("Finger Gait Example")

    ax[1].plot(time_x[:-1], delta_sol[:, 0, 6], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 7], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 8], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 12], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 13], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 14], label=r"$\gamma_1$")
    ax[1].legend()
    ax[1].set_ylabel(r"Gripper 1")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(time_x[:-1], delta_sol[:, 0, 9], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 10], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 11], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 15], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 16], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 17], label=r"$\gamma_2$")
    ax[2].legend()
    ax[2].set_ylabel(r"Gripper 2")
    ax[2].set_xlabel("Time (s)")

    ax[3].plot(time_x[:-1], u_sol)
    ax[3].legend(["G1 acceleration", 'G2 acceleration', 'G1 normal force', 'G2 normal force'])
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Input")
    # ax[3]
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
