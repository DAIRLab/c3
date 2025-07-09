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
    N = N
    h = 0.1
    g = 9.81
    mu = 1.0
    dt = 0.01
    A = np.array(
        [
            [1, h, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, h, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, h],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    B = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [h * h, 0, 0, 0],
            [h, 0, 0, 0],
            [0, h * h, 0, 0],
            [0, h, 0, 0],
        ]
    )
    D = np.array(
        [
            [0, h * h, -h * h, 0, h * h, -h * h],
            [0, h, -h, 0, h, -h],
            [0, -h * h, h * h, 0, 0, 0],
            [0, -h, h, 0, 0, 0],
            [0, 0, 0, 0, -h * h, h * h],
            [0, 0, 0, 0, -h, h],
        ]
    )
    E = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, -1],
            [0, -1, 0, 0, 0, 1],
        ]
    )
    F = np.array(
        [
            [0, -1, -1, 0, 0, 0],
            [1, 2 * h, -2 * h, 0, h, -h],
            [1, -2 * h, 2 * h, 0, -h, h],
            [0, 0, 0, 0, -1, -1],
            [0, h, -h, 1, 2 * h, -2 * h],
            [0, -h, h, 1, -2 * h, 2 * h],
        ]
    )
    c = np.array([[0], [-h * g], [h * g], [0], [-h * g], [h * g]])
    d = np.array([[-g * h * h], [-g * h], [0], [0], [0], [0]])
    H = np.array(
        [
            [0, 0, mu, 0],
            [-h, 0, 0, 0],
            [h, 0, 0, 0],
            [0, 0, 0, mu],
            [0, -h, 0, 0],
            [0, h, 0, 0],
        ]
    )
    # G = np.eye(n+m+k)
    return LCS(A, B, D, d, E, F, H, c, N, dt)


def make_fingergait_costs(lcs: LCS) -> ImprovedC3CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()
    print(f"Number of states: {n}, Number of lambdas: {m}, Number of inputs: {k}")

    R = [np.eye(k) for _ in range(N)]
    Q_init = np.array(
        [
            [1000, 0, 0, 0, 0, 0],
            [0, 20, 0, 0, 0, 0],
            [0, 0, 10, 0, 0, 0],
            [0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0],
            [0, 0, 0, 0, 0, 10],
        ]
    )
    Q = [Q_init for _ in range(N + 1)]
    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = np.diag(
        np.array([50, 50, 50, 50, 50, 50])
    )
    Ginit[n : n + m, n : n + m] = np.diag(np.array([50, 50, 50, 50, 50, 50]))
    G = [Ginit for _ in range(N)]

    scale = 6**2
    U = np.zeros((n + 2 * m + k, n + 2 * m + k))
    U[n : n + m, n : n + m] = np.diag(
        np.array([scale, scale, scale, scale, scale, scale])
    )
    U[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = np.eye(m)
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
    (gripper_marker1_left,) = ax.plot(
        [], [], "ro", markersize=6
    )  # red circle for gripper
    (gripper_marker1_right,) = ax.plot(
        [], [], "ro", markersize=6
    )  # red circle for gripper
    (gripper_marker2_left,) = ax.plot(
        [], [], "ro", markersize=6
    )  # red circle for gripper
    (gripper_marker2_right,) = ax.plot(
        [], [], "ro", markersize=6
    )  # red circle for gripper
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
        return (
            rod_line,
            limit_line,
            com_marker,
            gripper_marker1_left,
            gripper_marker1_right,
            gripper_marker2_left,
            gripper_marker2_right,
            time_text,
        )

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
        return (
            rod_line,
            com_marker,
            gripper_marker1_left,
            gripper_marker1_right,
            gripper_marker2_left,
            gripper_marker2_right,
            time_text,
        )

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=x.shape[1], interval=1000 / 60, blit=True
    )  # 60 FPS

    # Save animation
    writer = animation.PillowWriter(fps=60)
    anim.save("/home/hienbui/git/c3/fingerGait_animation.gif", writer=writer)

    return anim


def main():
    N = 10
    finger = make_fingergait_dynamics(N)
    costs = make_fingergait_costs(finger)

    n = finger.num_states()
    Xd0 = np.array([[0], [0], [0], [0], [6], [0]])
    xd = [Xd0 for _ in range(N + 1)]
    options = C3Options()
    options.admm_iter = 10
    options.rho_scale = 1.2
    options.num_threads = 10
    options.delta_option = 0

    opt = ImprovedC3(finger, costs, xd, options)

    x0 = np.array([[-8], [0], [1], [0], [5], [0]])

    system_iter = 100

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

    debug_delta = []

    u1 = np.array([0, 0, 1, 0])
    u2 = np.array([0, 0, 0, 1])
    debug_info = []
    debug_proj = []
    opt.AddLinearConstraint(a1, 1, 3, 1)
    opt.AddLinearConstraint(a2, 3, 5, 1)
    opt.AddLinearConstraint(u1, 0, 50, 2)
    opt.AddLinearConstraint(u2, 0, 50, 2)
    predict = []
    for i in range(system_iter):
        # whe i = 0; get result from SolveQP
        start_time = time.perf_counter()

        opt.Solve(x[:, i])
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        delta_sol.append(opt.GetDualDeltaSolution())
        u_opt = opt.GetInputSolution()[0]
        z_sol.append(opt.GetFullSolution())
        prediction = finger.Simulate(x[:, i], u_opt)
        predict.append(prediction)
        u_sol.append(u_opt)
        x[:, i + 1] = prediction
        x_.append(opt.GetStateSolution())
        # if i == system_iter - 1:
        #     breakpoint()
    sdf_sol = np.array(sdf_sol)
    delta_sol = np.array(delta_sol)

    dt = finger.dt()

    debug_info = opt.GetDebugInfo()
    debug_proj = opt.GetQPInfo()

    debug_info = np.array(debug_info)
    debug_proj = np.array(debug_proj)
    print("debug qp shape: ", debug_proj.shape)
    print("debug info shape: ", debug_info.shape)
    # print(debug_info.shape)
    # # save debug info to file
    z_sol = np.array(z_sol)
    predict = np.array(predict)
    # print(predict.shape)
    # print(x.shape)
    # Save the results to a file
    stored_folder = "/home/hienbui/git/c3/debug_output"
    np.save(f"{stored_folder}/debug.npy", debug_info)
    np.save(f"{stored_folder}/debug_projection.npy", debug_proj)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/z_sol.npy', z_sol)
    np.save(f"{stored_folder}/delta_sol.npy", delta_sol)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/x_.npy', x_)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/predict.npy', predict)

    # regular data
    # np.save("/home/yufeiyang/Documents/c3/debug_output/finger_x.npy", x)
    # np.save("/home/yufeiyang/Documents/c3/debug_output/finger_delta.npy", delta_sol)
    # np.save("/home/yufeiyang/Documents/c3/debug_output/finger_u.npy", u_sol)

    time_x = np.arange(0, system_iter * dt + dt, dt)

    u_sol = np.array(u_sol)
    print(u_sol.shape)

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    ax1.plot(time_x, x[0], color="r")
    ax1.plot(time_x, x[2], color="b", linestyle="-.")
    ax1.plot(time_x, x[4], color="k", linestyle="--")

    # Add shaded regions based on u_sol conditions
    time_u = time_x[:-1]  # u_sol has one fewer element than time_x

    # Get the y-axis limits for shading height
    y_min, y_max = ax1.get_ylim()
    y_range = y_max - y_min
    half_height = y_range / 2

    # Blue shading for u_sol[2] > 0 (top half)
    blue_mask = u_sol[:, 2] > 0
    blue_shaded = False
    if np.any(blue_mask):
        blue_shaded = True
        blue_regions = []
        start_idx = None
        for i, is_positive in enumerate(blue_mask):
            if is_positive and start_idx is None:
                start_idx = i
            elif not is_positive and start_idx is not None:
                blue_regions.append((time_u[start_idx], time_u[i - 1]))
                start_idx = None
        if start_idx is not None:
            blue_regions.append((time_u[start_idx], time_u[-1]))

        for start_time, end_time in blue_regions:
            ax1.axvspan(
                start_time, end_time, ymin=0.5, ymax=1.0, alpha=0.3, color="blue"
            )

    # Pink shading for u_sol[3] > 0 (bottom half)
    pink_mask = u_sol[:, 3] > 0
    pink_shaded = False
    if np.any(pink_mask):
        pink_shaded = True
        pink_regions = []
        start_idx = None
        for i, is_positive in enumerate(pink_mask):
            if is_positive and start_idx is None:
                start_idx = i
            elif not is_positive and start_idx is not None:
                pink_regions.append((time_u[start_idx], time_u[i - 1]))
                start_idx = None
        if start_idx is not None:
            pink_regions.append((time_u[start_idx], time_u[-1]))

        for start_time, end_time in pink_regions:
            ax1.axvspan(
                start_time, end_time, ymin=0.0, ymax=0.5, alpha=0.3, color="pink"
            )

    # Create legend labels
    legend_labels = [
        "Object position",
        "Gripper 1 position",
        "Gripper 2 position",
    ]

    # Get the plot lines for legend
    lines = ax1.get_lines()
    legend_handles = list(lines)

    # Add shaded region labels if they exist
    if blue_shaded:
        from matplotlib.patches import Patch

        blue_patch = Patch(color="blue", alpha=0.3, label="Gripper 1 normal force > 0")
        legend_handles.append(blue_patch)
        legend_labels.append("Gripper 1 normal force > 0")

    if pink_shaded:
        from matplotlib.patches import Patch

        pink_patch = Patch(color="pink", alpha=0.3, label="Gripper 2 normal force > 0")
        legend_handles.append(pink_patch)
        legend_labels.append("Gripper 2 normal force > 0")

    ax1.legend(legend_handles, legend_labels)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("State")
    ax1.set_title("Finger Gait Example")

    # ax1[1].plot(time_x[:-1], u_sol)
    # ax1[1].legend(
    #     ["G1 acceleration", "G2 acceleration", "G1 normal force", "G2 normal force"]
    # )
    # ax1[1].set_xlabel("Time (s)")
    # ax1[1].set_ylabel("Input")

    # fig2, ax2 = plt.subplots(6, 1, figsize=(10, 10))
    # ax2[0].plot(time_x[:-1], delta_sol[:, 0, 6], label=r"$Slack var$")
    # ax2[0].plot(time_x[:-1], delta_sol[:, 0, 12], label=r"$\gamma_1$")
    # ax2[0].legend(["Gripper 1 Slack var", "gamma"])

    # ax2[1].plot(time_x[:-1], delta_sol[:, 0, 7], label=r"$Pos tangential force$")
    # ax2[1].plot(time_x[:-1], delta_sol[:, 0, 13], label=r"$\gamma_1$")
    # ax2[1].legend(["Gripper 1 Pos tangential force", "gamma"])

    # ax2[2].plot(time_x[:-1], delta_sol[:, 0, 8], label=r"$Neg tangential force$")
    # ax2[2].plot(time_x[:-1], delta_sol[:, 0, 14], label=r"$\gamma_1$")
    # ax2[2].legend(["Gripper 1 Neg tangential force", "gamma"])

    # ax2[3].plot(time_x[:-1], delta_sol[:, 0, 9], label=r"$Slack var$")
    # ax2[3].plot(time_x[:-1], delta_sol[:, 0, 15], label=r"$\gamma_2$")
    # ax2[3].legend(["Gripper 2 Slack var", "gamma"])

    # ax2[4].plot(time_x[:-1], delta_sol[:, 0, 10], label=r"$Pos tangential force$")
    # ax2[4].plot(time_x[:-1], delta_sol[:, 0, 16], label=r"$\gamma_2$")
    # ax2[4].legend(["Gripper 2 Pos tangential force", "gamma"])

    # ax2[5].plot(time_x[:-1], delta_sol[:, 0, 11], label=r"$Neg tangential force$")
    # ax2[5].plot(time_x[:-1], delta_sol[:, 0, 17], label=r"$\gamma_2$")
    # ax2[5].legend(["Gripper 2 Neg tangential force", "gamma"])

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("/home/hienbui/git/c3/debug_output/finger_gait.png", dpi=600)


if __name__ == "__main__":
    main()
