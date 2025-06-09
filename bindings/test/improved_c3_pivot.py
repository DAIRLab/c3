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


def make_pivot_dynamics(N: int) -> LCS:
    g = 9.81
    dt = 0.01
    mu1 = 0.1
    mu2 = 9.81
    mu3 = 1.0
    h = 1
    w = 1
    mm = 1
    xcurrent = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    sinn = np.sin(xcurrent[4]).item()
    coss = np.cos(xcurrent[4]).item()
    x6 = xcurrent[6, 0]
    x8 = xcurrent[8, 0]
    x5 = xcurrent[5, 0]
    x7 = xcurrent[7, 0]
    rt = (-1) * np.sqrt(h * h + w * w)
    z = w * sinn - h*coss
    A = np.array(
        [
            [1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    )
    B = np.array([[0, 0, -dt * dt * coss, dt * dt * sinn],
                  [0, 0, -dt * coss, dt * sinn],
                  [0, 0, dt * dt * sinn, dt * dt * coss],
                  [0, 0, dt * sinn, dt * coss],
                  [0, 0, -dt * dt * x6, dt * dt * x8],
                  [0, 0, -dt * x6, dt * x8],
                  [dt * dt, 0, 0, 0],
                  [dt, 0, 0, 0],
                  [0, dt * dt, 0, 0],
                  [0, dt, 0, 0]])

    D = np.array([
        [0, dt * dt * sinn, -dt * dt * sinn, 0, -coss * dt * dt, dt * dt * coss, 0, dt * dt, -dt * dt, 0],
        [0, dt * sinn, -dt * sinn, 0, -coss * dt, dt * coss, 0, dt, -dt, 0],
        [0, dt * dt * coss, -dt * dt * coss, 0, dt * dt * sinn, -dt * dt * sinn, 0, 0, 0, dt * dt],
        [0, dt * coss, -dt * coss, 0, dt * dt * sinn, -dt * sinn, 0, 0, 0, dt],
        [0, -dt * dt * h, dt * dt * h, 0, dt * dt * w, -dt * dt * w, 0, -dt * dt * coss * w - dt * dt * sinn * h,
        dt * dt * coss * w + dt * dt * sinn * h, dt * dt * sinn * w - h * coss * dt * dt],
        [0, -dt * h, dt * h, 0, dt * w, -dt * w, 0, -dt * coss * w - dt * sinn * h,
        dt * coss * w + dt * sinn * h, dt * sinn * w - h * coss * dt],
        [0, -dt * dt, dt * dt, 0, 0, 0, 0, 0, 0, 0],
        [0, -dt, dt, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -dt * dt, dt * dt, 0],
        [0, 0, 0, 0, 0, 0, 0, -dt, dt, 0]
    ])
    E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -1, 0, 0, 0, -rt, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, rt, 0, 0, 0, 0],
                  [0, 0, 1, dt,-h * sinn + w * coss + z, z * dt, 0, 0, 0, 0]])
    F = np.array([
        [0,   -1,   -1,     0,    0,    0,    0,     0,     0,     0],
        [0,    1,   dt,   -dt,    0,    0,    0,     0,     0,     0],
        [0,    1,  -dt,    dt,    0,    0,    0,     0,     0,     0],
        [0,    0,    0,     0,   -1,   -1,    0,     0,     0,     0],
        [0,    0,    1,    dt,  -dt,    0,    0,     0,     0,     0],
        [0,    0,    1,   -dt,   dt,    0,    0,     0,     0,     0],
        [0,    0,    0,     0,   -1,   -1,  mu3,     0,  -(dt * sinn - rt * dt * h), -(-dt * sinn + rt * dt * h)],
        [0,    0,    0,     0,    0,    0,     -(-dt * coss + rt * dt * w), -(dt * coss - dt * rt * w), -(-1), -(dt - rt * dt * coss * w - rt * dt * sinn * h)],
        [0,    0,    0,     0,    0,    0,     -(-dt + rt * dt * coss * w + rt * dt * sinn * h), -(sinn * w * rt * dt - h * coss * rt * dt), 0, -(-dt * sinn + rt * dt * h)],
        [-(dt * sinn - rt * dt * h), 0, -(dt * coss - rt * dt * w), -(-dt * coss + rt * dt * w), -(-1), -(-dt + rt * dt * coss * w + rt * dt * sinn * h),
        (dt - rt * dt * coss * w - rt * dt * sinn * h), -(-sinn * w * rt * dt + h * coss * rt * dt), 0, dt * dt * coss - z * dt * dt * h]
    ])
    
    c = np.array([[0], [-h * g], [h * g], [0], [-h *g], [h * g], [0], [0], [0], [0]])
    d = np.array([[0], [0], [-dt * dt * mm * g], [-dt * mm * g], [0], [0], [0], [0], [0], [0]])
    H = np.array([
        [0, 0, mu1, 0],
        [-dt, 0, 0, 0],
        [dt, 0, 0, 0],
        [0, 0, 0, 0],
        [0, mu2, 0, -dt],
        [0, 0, 0, dt],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [-(-dt * coss - rt * dt * x5), -(dt * sinn + dt * rt * x7), 0, 0],
        [-(dt * coss + rt * dt * x5), -(-dt * sinn - dt * rt * x7), 0, 0],
        [dt * dt * sinn - dt * dt * x5 * z, dt * dt * coss + dt * dt * x7 * z, 0, 0]
    ])
    return LCS(A, B, D, d, E, F, H, c, N, dt)


def make_pivot_costs(lcs: LCS) -> ImprovedC3CostMatrices:
    N = lcs.N()
    n = lcs.num_states()
    m = lcs.num_lambdas()
    k = lcs.num_inputs()
    print(f"Number of states: {n}, Number of lambdas: {m}, Number of inputs: {k}")

    R = [0.01*np.eye(k) for _ in range(N)]
    Qinit = np.eye(n)
    Qinit[4, 4] = 100
    Qinit[2, 2] = 100
    Qinit[0, 0] = 100
    Qinit[6, 6] = 50
    Qinit[8, 8] = 50
    Qinit[5, 5] = 11
    Qinit[3, 3] = 9
    Qinit[1, 1] = 11

    Q = [Qinit for _ in range(N + 1)]

    Ginit = np.zeros((n + 2 * m + k, n + 2 * m + k))
    Ginit[n + m + k : n + 2 * m + k, n + m + k : n + 2 * m + k] = np.eye(m)
    Ginit[n : n + m, n : n + m] = np.eye(m)
    # Ginit = np.eye(n + 2 * m + k)
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
        pole_x = [cart_pos, cart_pos - len_p * np.sin(pole_angle)]
        pole_y = [0, len_p * np.cos(pole_angle)]

        # Calculate center of mass position
        com_x = cart_pos - len_com * np.sin(pole_angle)
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
    anim.save("/home/yufeiyang/Documents/c3/cartpole_animation.gif", writer=writer)

    return anim


def main():
    N = 10
    cartpole = make_pivot_dynamics(N)
    costs = make_pivot_costs(cartpole)

    n = cartpole.num_states()

    x0 = np.array([0, 0, 1.36, 0, 0.2, 0, -0.3, 0, -0.7, 0]).reshape(-1, 1)
    xd = [np.zeros((n, 1)) for _ in range(N + 1)]
    options = C3Options()
    options.admm_iter = 10
    options.rho_scale = 2
    options.num_threads = 10
    options.delta_option = 0

    opt = ImprovedC3(cartpole, costs, xd, options)

    # x0 = np.array([0, -0.5, 0.5, -0.4])

    system_iter = 1000

    x = np.zeros((n, system_iter + 1))

    x[:, 0] = x0.ravel()
    solve_times = []
    sdf_sol = []
    delta_sol = []
    z_sol = []
    x_ = []
    u_sol = []
    # debug_info = []
    for i in range(system_iter):
        # whe i = 0; get result from SolveQP
        start_time = time.perf_counter()
        opt.Solve(x[:, i])
        if i == 100:
            debug_info = opt.GetDebugInfo()
        solve_times.append(time.perf_counter() - start_time)
        sdf_sol.append(opt.GetSDFSolution())
        delta_sol.append(opt.GetDualDeltaSolution())
        u_opt = opt.GetInputSolution()[0]
        z_sol.append(opt.GetFullSolution())
        prediction = cartpole.Simulate(x[:, i], u_opt)
        u_sol.append(u_opt)
        x[:, i + 1] = prediction
        x_.append(opt.GetStateSolution())


    sdf_sol = np.array(sdf_sol)
    delta_sol = np.array(delta_sol)
    print(delta_sol.shape)

    dt = cartpole.dt()

    debug_info = np.array(debug_info)
    print(debug_info.shape)
    # save debug info to file
    with open('/home/yufeiyang/Documents/c3/debug_output/debug_info.txt', 'a') as f:
        f.write('\n')
        np.savetxt(f, debug_info[1], fmt='%s')

    z_sol = np.array(z_sol)
    print(z_sol.shape)
    # # Save the results to a file
    # np.save('/home/yufeiyang/Documents/c3/debug_output/debug.npy', debug_info)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/z_sol.npy', z_sol)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/delta_sol.npy', delta_sol)
    # np.save('/home/yufeiyang/Documents/c3/debug_output/x_.npy', x_)

    # Create animation if necessary
    len_p = 0.6  # pole length
    len_com = 0.4267  # center of mass length
    # anim = animate_cartpole(x, dt, len_p, len_com)

    time_x = np.arange(0, system_iter * dt + dt, dt)

    print(
        f"Average solve time: {np.mean(solve_times)}, equivalent to {1 / np.mean(solve_times)} Hz"
    )
    u_sol = np.array(u_sol)
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))

    ax[0].plot(time_x, x.T)
    # ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("State")
    ax[0].set_title("Pivot Example")

    ax[1].plot(time_x[:-1], delta_sol[:, 0, 10], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 11], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 12], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 13], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 14], label=r"$\lambda_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 20], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 21], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 22], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 23], label=r"$\gamma_1$")
    ax[1].plot(time_x[:-1], delta_sol[:, 0, 24], label=r"$\gamma_1$")
    ax[1].legend()
    ax[1].set_ylabel(r"Right Wall")
    ax[1].set_xlabel("Time (s)")

    ax[2].plot(time_x[:-1], delta_sol[:, 0, 15], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 16], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 17], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 18], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 19], label=r"$\lambda_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 25], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 26], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 27], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 28], label=r"$\gamma_2$")
    ax[2].plot(time_x[:-1], delta_sol[:, 0, 29], label=r"$\gamma_2$")
    ax[2].legend()
    ax[2].set_ylabel(r"Left Wall")
    ax[2].set_xlabel("Time (s)")

    ax[3].plot(time_x[:-1], u_sol)
    # ax[3].legend(["G1 acceleration", 'G2 acceleration', 'G1 normal force', 'G2 normal force'])
    ax[3].set_xlabel("Time (s)")
    ax[3].set_ylabel("Input")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
