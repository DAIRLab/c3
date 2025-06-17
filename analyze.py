import matplotlib.pyplot as plt
import numpy as np

def make_plot(arr, name):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(arr, label=f"${name}$", marker='o')
    plt.xlabel("ADMM iteration")
    plt.ylabel("Value")
    plt.title(f"ADMM iteratin vs {name}$")
    plt.legend()
    plt.grid()
    plt.tight_layout()

delta = np.load('/home/yufeiyang/Documents/c3/debug_output/debug_qp.npy')
 #This contains the delta value after 10 ADMM in the first timestep

print(delta.shape)
# ADMM iter by N by (n_x + 2*n_lambda + n_u)
# choose one time step
curr_delta = delta[-20:,0, :]

# plot the curr_delta
x_1 = curr_delta[:, 0]
x_2 = curr_delta[:, 1]
x_3 = curr_delta[:, 2]
x_4 = curr_delta[:, 3]
x_5 = curr_delta[:, 4]
x_6 = curr_delta[:, 5]
lambda_11 = curr_delta[:, 6]
lambda_12 = curr_delta[:, 7]
lambda_13 = curr_delta[:, 8]

gamma_11 = curr_delta[:, 9]
gamma_12 = curr_delta[:, 10]
gamma_13 = curr_delta[:, 11]

lambda_21 = curr_delta[:, 12]
lambda_22 = curr_delta[:, 13]
lambda_23 = curr_delta[:, 14]

gamma_21 = curr_delta[:, 15]
gamma_22 = curr_delta[:, 16]
gamma_23 = curr_delta[:, 17]

u1 = curr_delta[:, 18]
u2 = curr_delta[:, 19]
u3 = curr_delta[:, 20]
u4 = curr_delta[:, 21]

make_plot(x_1, "x1 Object position")
make_plot(x_2, "x2 Object velocity")
make_plot(x_3, "x3 Gripper 1 position")
make_plot(x_4, "x4 Gripper 1 velocity")
make_plot(x_5, "x5 Gripper 2 position")
make_plot(x_6, "x6 Gripper 2 velocity")
make_plot(lambda_11, "Gripper 1 slack variable")
make_plot(lambda_12, "Gripper 1 pos tangential force")
make_plot(lambda_13, "Gripper 1 neg tangential force")
make_plot(gamma_11, "gamma_11")
make_plot(gamma_12, "gamma_12")  
make_plot(gamma_13, "gamma_13")
make_plot(lambda_21, "gripper 2 slack variable")
make_plot(lambda_22, "gripper 2 pos tangential force")
make_plot(lambda_23, "gripper 2 neg tangential force")
make_plot(gamma_21, "gamma_21")
make_plot(gamma_22, "gamma_22")
make_plot(gamma_23, "gamma_23")
make_plot(u1, "u1")
make_plot(u2, "u2")
make_plot(u3, "u3")
make_plot(u4, "u4")
# plt.show()       

qp_data = delta = np.load('/home/yufeiyang/Documents/c3/debug_output/debug_qp.npy')
qp_data = qp_data[0, :, :]
qp_x1 = qp_data[:, 0]
qp_x2 = qp_data[:, 1]
qp_x3 = qp_data[:, 2]
qp_x4 = qp_data[:, 3]
qp_x5 = qp_data[:, 4]

# make_plot(qp_x1, "qp_x1")
# make_plot(qp_x2, "qp_x2")
# make_plot(qp_x3, "qp_x3")
# make_plot(qp_x4, "qp_x4")
# make_plot(qp_x5, "qp_x5")




plt.show()