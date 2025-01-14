import numpy as np
from system_dynamics import*
from system_costs import*
from C3 import*
import matplotlib.pyplot as plt
from pyc3 import LCS, C3MIQP, C3Options

#Parameters of the linear complementarity system [ x_{k+1} = A x_k + B u_k + D lam_k + d, 0 <= lam \perp E x_k + F lam_k + H u_k >= 0 ]
#[ n = state vector dimensions, m = contact vector dimension, k = input vector dimension ]
#[ A in R^{n times n}, B in R^{n times k}, D in R^{n times m}, d in R^{n}, E in R^{m times n}, F in R^{m times m}, H in R^{m times k}, c in R^{m} ]
A, B, D, d, E, c, F, H, n, m, k = system_dynamics()

#Parameters of the cost [ \sum_k x_k^T Q x_k + u_k^T R u_k + x_N^T Q_N x_N ]
#[ Q in R^{n times n}, R in R^{k times k}, Q_N in R^{n times n} ]
#[ x0 = initial condition (in R^{n}, N = MPC horizon (int), x_des = goal (desired) state (in R^{n}) ]
Q, R, G, U, x_des, QN, x0, N = system_costs(A, B)

#Create the LCS
cartpole = LCS(A,B,D,d,E,F,H,c,N)

#Options for the C3 Solver
options = C3Options()
options.admm_iter = 10 #Number of ADMM iterations
options.rho_scale = 2 #Scaling for the rho parameter
options.OSQP_verbose_flag = 0 #Verbose flag for OSQP solver (0: suppress, 1: print out)
options.OSQP_maximum_iterations = 50 #Maximum number of iterations for OSQP solver
options.Gurobi_verbose_flag = 0 #Verbose flag for Gurobi solver (0: suppress, 1: print out)
options.Gurobi_maximum_iterations = 10 #Maximum number of iterations for Gurobi solver
delta_option = 1 #Initialization of delta, 0 for initializing as zero vector, 1 for initializing with the current state

#Create the C3 optimizer
opt = C3MIQP.MakeTimeInvariantC3MIQP(cartpole, Q, R, G, U, x_des, QN, options, N)

system_iter = 500 #Number of simulation steps

#State information is saved on x
x = np.zeros((n, system_iter+1))
x[:, [0]]  = x0

for i in range(system_iter):

	#Compute u_0 given x_0 using the C3 optimizer
	input = C3(x[:, [i]], opt, n, m, k, N, delta_option)

	#Compute the next state given x_0 and u_0 (save the value on x)
	state_next = cartpole.Simulate(x[:, [i]], input)
	x[:, [i+1]] = np.reshape(state_next, (n,1))

	#Solver progress
	if i % 50 == 0:
		print(i,"/500")

#Plot the state evolution x
dt = 0.01
time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x.T)
plt.show()

