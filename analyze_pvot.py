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

delta = np.load('/home/yufeiyang/Documents/c3/debug_output/debug_pivot.npy')
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
x_7 = curr_delta[:, 6]
x_8 = curr_delta[:, 7]
x_9 = curr_delta[:, 8]
x_10 = curr_delta[:, 9]
lambda_11 = curr_delta[:, 10]
lambda_12 = curr_delta[:, 11]
lambda_13 = curr_delta[:, 12]
lambda_14 = curr_delta[:, 13]
lambda_15 = curr_delta[:, 14]
lambda_16 = curr_delta[:, 15]
lambda_17 = curr_delta[:, 16]
lambda_18 = curr_delta[:, 17]
lambda_19 = curr_delta[:, 18]
lambda_20 = curr_delta[:, 19]

gamma_11 = curr_delta[:, 20]
gamma_12 = curr_delta[:, 21]
gamma_13 = curr_delta[:, 22]
gamma_14 = curr_delta[:, 23]
gamma_15 = curr_delta[:, 24]
gamma_16 = curr_delta[:, 25]
gamma_17 = curr_delta[:, 26]
gamma_18 = curr_delta[:, 27]
gamma_19 = curr_delta[:, 28]
gamma_20 = curr_delta[:, 29]

# lambda_21 = curr_delta[:, 12]
# lambda_22 = curr_delta[:, 13]
# lambda_23 = curr_delta[:, 14]

# gamma_21 = curr_delta[:, 15]
# gamma_22 = curr_delta[:, 16]
# gamma_23 = curr_delta[:, 17]

# u1 = curr_delta[:, 18]
# u2 = curr_delta[:, 19]
# u3 = curr_delta[:, 20]
# u4 = curr_delta[:, 21]

# make_plot(x_1, "COM x")
# make_plot(x_2, "COM x vel")
# make_plot(x_3, "COM y")
# make_plot(x_4, "COM y vel")
# make_plot(x_5, "object angle")
# make_plot(x_6, "object angle vel")
# make_plot(x_7, "finger 1 pos")
# make_plot(x_8, "finger 1 vel")
# make_plot(x_9, "finger 2 pos")
# make_plot(x_10, "finger 2 vel")
# make_plot(lambda_11, "fnger 1 slack variable")
# make_plot(lambda_12, "fnger 1 friction 1")
# make_plot(lambda_13, "fnger 1 friction 2")
# make_plot(lambda_14, "fnger 2 slack variable")
# make_plot(lambda_15, "fnger 2 friction 1")
# make_plot(lambda_16, "fnger 2 friction 2")
# make_plot(lambda_17, "Ground normal force")
# make_plot(lambda_18, "Ground slack variable")
# make_plot(lambda_19, "Ground friction 1")
# make_plot(lambda_20, "Ground friction 2")
# make_plot(gamma_11, "gamma_11")
# make_plot(gamma_12, "gamma_12")  
# make_plot(gamma_13, "gamma_13")
# make_plot(gamma_14, "gamma_14")
# make_plot(gamma_15, "gamma_15")
# make_plot(gamma_16, "gamma_16")
# make_plot(gamma_17, "gamma_17")
# make_plot(gamma_18, "gamma_18")
# make_plot(gamma_19, "gamma_19")
# make_plot(gamma_20, "gamma_20")
# plt.show()       

qp_data = np.load('/home/yufeiyang/Documents/c3/debug_output/debug_qp.npy')
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