import matplotlib.pyplot as plt
import numpy as np
import click


def make_plot(arr_list, name_list, title, ax=None, label="ADMM Iteration"):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_fig = True

    for arr, name in zip(arr_list, name_list):
        ax.plot(arr, label=name)

    ax.set_xlabel(label)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(fontsize=14)
    ax.grid()


def plot_QP(horizon, timestep, admm_num, delta, title, plot_horizon):
    #  This is for the QP output before projection step in each ADMM
    # delta = np.load('/home/yufeiyang/Documents/c3/debug_output/debug.npy')
    admm_iter = admm_num
    admm_time, N, num_var = delta.shape
    delta = delta.reshape(-1, admm_iter, N, num_var)   # system iter, ADMM, N, num_var

    print(delta.shape)
    if plot_horizon:
        curr_delta = delta[timestep, -1, :, :]
        label = "Planned horizon"
    else:
        curr_delta = delta[timestep, :, horizon, :]
        label = "ADMM iteration"


    # plot the curr_delta
    x_1 = curr_delta[:, 0]  # object pos
    x_2 = curr_delta[:, 1] # object vel
    x_3 = curr_delta[:, 2] # gripper 1 po
    x_4 = curr_delta[:, 3] # gripper 1 vel
    x_5 = curr_delta[:, 4] # gripper 2 pos
    x_6 = curr_delta[:, 5] # gripper 2 vel
    lambda_11 = curr_delta[:, 6] # gripper 1 maximum sliding vel
    print(lambda_11)
    lambda_12 = curr_delta[:, 7] # gripper 1 pos tangential friction on the object
    lambda_13 = curr_delta[:, 8] # gripper 1 neg tangential friction on the object

    lambda_21 = curr_delta[:, 9]
    lambda_22 = curr_delta[:, 10]
    lambda_23 = curr_delta[:, 11]

    u1 = curr_delta[:, 12] # gripper 1 accel
    u2 = curr_delta[:, 13] # gripper 2 accel
    u3 = curr_delta[:, 14] # gripper 1 normal force 
    u4 = curr_delta[:, 15] # gripper 2 normal force

    gamma_11 = curr_delta[:, 16]
    gamma_12 = curr_delta[:, 17]
    gamma_13 = curr_delta[:, 18]


    gamma_21 = curr_delta[:, 19]
    gamma_22 = curr_delta[:, 20]
    gamma_23 = curr_delta[:, 21]


    fig1, ax1 = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)
    make_plot([x_1, x_3, x_5], ["x1 Object position", "x3 Gripper 1 position", "x5 Gripper 2 position"], "States position", ax=ax1[0], label=label)
    make_plot([x_2, x_4, x_6], ["x2 Object velocity", "x4 Gripper 1 velocity", "x6 Gripper 2 velocity"], "States velocity", ax=ax1[1], label=label)
    make_plot([u1, u2], ["Gripper 1 acceleration", "Gripper 2 acceleration"], "Input cceleration",ax=ax1[2], label=label)
    make_plot([u3, u4], ["Gripper 1 normal force", "Gripper 2 normal force"], "Input normal force",ax=ax1[3], label=label)
    if plot_horizon:
        fig1.suptitle(f"Timestep {timestep} planned horizon", fontsize=16)
    else:    
        fig1.suptitle(f"{title} at timestep {timestep}, planning step {horizon}", fontsize=16)

    fig2, ax2 = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)
    make_plot([lambda_11, lambda_21], ["Gripper 1 maximum sliding velocity", "gripper 2 maximum sliding velocity"], "Grippers lambda slack variable",ax=ax2[0], label=label)
    make_plot([gamma_11, gamma_21], ["Gripper 1 friction cone", 'Gripper 2 friction cone'], "slack variable complementarity", ax=ax2[1], label=label)
    make_plot([lambda_12, lambda_13, lambda_22, lambda_23], ["Gripper 1 pos tangential force", "Gripper 1 neg tangential force", "gripper 2 pos tangential force", "gripper 2 neg tangential force"], "Grippers lambda friction forces",ax=ax2[2], label=label)
    make_plot([gamma_12, gamma_13, gamma_22, gamma_23], ["Gripper 1 pos vel", "Gripper 1 neg vel", "Gripper 2 pos vel", "Gripper 2 neg vel"], "Friction force complementarity",ax=ax2[3], label=label)  
    if plot_horizon:
        fig2.suptitle(f"Timestep {timestep} planned horizon", fontsize=16)
    else:
        fig2.suptitle(f"{title} at timestep {timestep}, planning step {horizon}", fontsize=16)


def plot_original(horizon, timestep, admm_num, delta, title, plot_horizon):
    #  This is for the QP output before projection step in each ADMM
    # delta = np.load('/home/yufeiyang/Documents/c3/debug_output/debug.npy')
    admm_iter = admm_num
    admm_time, N, num_var = delta.shape
    delta = delta.reshape(-1, admm_iter, N, num_var)   # system iter, ADMM, N, num_var

    print(delta.shape)
    
    if plot_horizon:
        curr_delta = delta[timestep, -1, :, :]
    else:
        curr_delta = delta[timestep, :, horizon, :]

    # plot the curr_delta
    x_1 = curr_delta[:, 0]  # object pos
    x_2 = curr_delta[:, 1] # object vel
    x_3 = curr_delta[:, 2] # gripper 1 po
    x_4 = curr_delta[:, 3] # gripper 1 vel
    x_5 = curr_delta[:, 4] # gripper 2 pos
    x_6 = curr_delta[:, 5] # gripper 2 vel
    lambda_11 = curr_delta[:, 6] # gripper 1 maximum sliding vel
    print(lambda_11)
    lambda_12 = curr_delta[:, 7] # gripper 1 pos tangential friction on the object
    lambda_13 = curr_delta[:, 8] # gripper 1 neg tangential friction on the object

    lambda_21 = curr_delta[:, 9]
    lambda_22 = curr_delta[:, 10]
    lambda_23 = curr_delta[:, 11]

    u1 = curr_delta[:, 12] # gripper 1 accel
    u2 = curr_delta[:, 13] # gripper 2 accel
    u3 = curr_delta[:, 14] # gripper 1 normal force 
    u4 = curr_delta[:, 15] # gripper 2 normal force


    fig1, ax1 = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)
    make_plot([x_1, x_3, x_5], ["x1 Object position", "x3 Gripper 1 position", "x5 Gripper 2 position"], "States position", ax=ax1[0])
    make_plot([x_2, x_4, x_6], ["x2 Object velocity", "x4 Gripper 1 velocity", "x6 Gripper 2 velocity"], "States velocity", ax=ax1[1])
    make_plot([u1, u2], ["Gripper 1 acceleration", "Gripper 2 acceleration"], "Input cceleration",ax=ax1[2])
    make_plot([u3, u4], ["Gripper 1 normal force", "Gripper 2 normal force"], "Input normal force",ax=ax1[3])

    fig1.suptitle(f"{title} at timestep {timestep}, planning step {horizon}", fontsize=16)
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    make_plot([lambda_11, lambda_21], ["Gripper 1 maximum sliding velocity", "gripper 2 maximum sliding velocity"], "Grippers lambda slack variable",ax=ax2[0])
    make_plot([lambda_12, lambda_13, lambda_22, lambda_23], ["Gripper 1 pos tangential force", "Gripper 1 neg tangential force", "gripper 2 pos tangential force", "gripper 2 neg tangential force"], "Grippers lambda friction forces",ax=ax2[1])
    fig2.suptitle(f"{title} at timestep {timestep}, planning step {horizon}", fontsize=16)

@click.command()
@click.option('--original', default=False, type=bool)
@click.option('--projection', default=0, type=int, help='0 for QP output before projection, 1 for delta after projection')
@click.option('--horizon', default=0, type=int, help='Horizon timestep')
@click.option('--timestep', default=-1, type=int)
@click.option('--admm_num', default=10, type=int, help='Number of ADMM iterations')
@click.option('--plot_horizon', default=False, type=bool)
def plot(original, projection, horizon, timestep, admm_num, plot_horizon):
    qp_data = np.load('/home/yufeiyang/Documents/c3/debug_output/debug.npy')  # QP
    delta_data = np.load('/home/yufeiyang/Documents/c3/debug_output/debug_projection.npy')  # after projection
    title_qp = "ADMM QP output before projection"
    title_delta = "ADMM projection output"
    if original:
        title_original_qp = "Original ADMM QP output before projection"
        title_original_delta = "Original ADMM projection output"
        qp_data = np.load("/home/yufeiyang/Documents/c3/debug_output/original_z.npy")
        delta_data = np.load("/home/yufeiyang/Documents/c3/debug_output/original_delta.npy")
        if projection == 0:
            plot_QP(horizon, timestep, admm_num, qp_data, title_original_qp, plot_horizon)
        elif projection == 1:
            plot_QP(horizon, timestep, admm_num, delta_data, title_original_delta, plot_horizon)
    else:
        if projection == 0:  # before projection
            plot_QP(horizon, timestep, admm_num, qp_data, title_qp, plot_horizon)
        elif projection == 1 or plot_horizon: # after projection
            plot_QP(horizon, timestep, admm_num, delta_data, title_delta, plot_horizon)
        else:
            plot_QP(horizon, timestep, admm_num, qp_data, title_qp, plot_horizon)
            plot_QP(horizon, timestep, admm_num, delta_data, title_delta, plot_horizon)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()