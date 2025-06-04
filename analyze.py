import matplotlib.pyplot as plt
import numpy as np


def plot_data(system_iter, x):
    # system_iter = 1000
    dt = 0.01
    time_x = np.arange(0, system_iter * dt + dt, dt)
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))

    ax[0].plot(time_x, x.T)
    ax[0].legend(["Cart Position", "Pole Angle", "Cart Velocity", "Pole Velocity"])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("State")
    ax[0].set_title("Improved C3 Controller")


    plt.show()

# load the file
# file_path = "/home/yufeiyang/Documents/c3/debug_output/z_sol.npy"
# z_sol = np.load(file_path) #N_ by (n_x+2*n_lambda+n_u_) 10 by 9
# delta_sol = np.load('/home/yufeiyang/Documents/c3/debug_output/delta_sol.npy')
x_sol = np.load('/home/yufeiyang/Documents/c3/debug_output/x_.npy')
system_iter = x_sol.shape[0] 
n_x = 4
n_lambda = 2
n_u = 1
print(x_sol.shape)

for i in range(x_sol.shape[0]):
    sol = x_sol[i] # 10 by 4
    plot_data(system_iter, sol)
