import numpy as np
from system_dynamics import*
from system_costs import*
import matplotlib.pyplot as plt
from pyc3 import LCS, C3MIQP, C3Options

#Parameters of the linear complementarity system [ x_{k+1} = A x_k + B u_k + D lam_k + d, 0 <= lam \perp E x_k + F lam_k + H u_k >= 0 ]
#[ n = state vector dimensions, m = contact vector dimension, k = input vector dimension ]
A, B, D, d, E, c, F, H, n, m, k = system_dynamics()

#Parameters of the cost [ \sum_k x_k^T Q x_k + u_k^T R u_k + x_N^T Q_N x_N ]
#[ x0 = initial condition, N = MPC horizon, x_des = goal (desired) state ]
Q, R, G, U, x_des, QN, x0, N = system_costs(A, B)

#Create the LCS
cartpole = LCS(A,B,D,d,E,F,H,c,N)

#Options for the C3 Solver
options = C3Options()
options.admm_iter = 10
options.rho_scale = 2
options.OSQP_verbose_flag = 0
options.OSQP_maximum_iterations = 50
options.Gurobi_verbose_flag = 0
options.Gurobi_maximum_iterations = 10

#Create the optimizer
opt = C3MIQP.MakeTimeInvariantC3MIQP(cartpole, Q, R, G, U, x_des, QN, options, N)


delta_add = np.zeros((n+m+k, 1))
w_add = np.zeros((n+m+k, 1))

system_iter = 500

x = np.zeros((n, system_iter+1))
x[:, [0]]  = x0

for i in range(system_iter):

	delta = []
	w = []

	for j in range(N):
		delta.append(delta_add)
		w.append(w_add)

	input = opt.Solve(x[:, [i]], delta, w)

	prediction = cartpole.Simulate(x[:, [i]], input)
	x[:, [i+1]] = np.reshape(prediction, (n,1))

	if i % 50 == 0:
		print(i)



dt = 0.01
time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x.T)
plt.show()

