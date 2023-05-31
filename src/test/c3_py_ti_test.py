import numpy as np
from scipy import linalg

from pyc3 import LCS, C3MIQP, C3Options

n=4
m=2
k=1
N = 10
g = 9.81
mp = 0.411
mc = 0.978
len_p = 0.6
len_com = 0.4267
d1 = 0.35
d2 = -0.35
ks= 100
Ts = 0.01
A = [[0, 0, 1, 0], [0, 0, 0, 1], [0, g*mp/mc, 0, 0], [0, g*(mc+mp)/(len_com*mc), 0, 0]]
A = np.asarray(A)
B = [[0],[0],[1/mc],[1/(len_com*mc)]]
B = np.asarray(B)
D = [[0,0], [0,0], [(-1/mc) + (len_p/(mc*len_com)), (1/mc) - (len_p/(mc*len_com)) ], [(-1 / (mc*len_com) ) + (len_p*(mc+mp)) / (mc*mp*len_com*len_com)  , -((-1 / (mc*len_com) ) + (len_p*(mc+mp)) / (mc*mp*len_com*len_com))    ]]
D = np.asarray(D)
E = [[-1, len_p, 0, 0], [1, -len_p, 0, 0 ]]
E = np.asarray(E)
F = 1/ks * np.eye(2)
F = np.asarray(F)
c = [[d1], [-d2]]
c = np.asarray(c)
d = np.zeros((4,1))
H = np.zeros((2,1))
A = np.eye(n) + Ts * A
B = Ts*B
D = Ts*D
d = Ts*d

Q = np.array([[10, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

R = np.array([[1]])

G = 0.1*np.identity(n+m+k)
G[6,6] = 0

px = 1000
plam = 1
pu = 0
U = [[px, 0, 0, 0, 0, 0, 0], [0, px, 0, 0, 0, 0, 0 ], [0, 0, px, 0, 0, 0, 0 ], [0, 0, 0, px, 0 ,0 ,0], [0, 0, 0, 0, plam, 0, 0], [0, 0, 0, 0, 0, plam, 0  ], [0,0,0,0,0,0,0]]
U= np.asarray(U)

x_des = np.zeros(n)

#change with algebraic ricatti
QN = linalg.solve_discrete_are(A, B, Q, R)

cartpole = LCS(A,B,D,d,E,F,H,c,N)

options = C3Options()

opt = C3MIQP.MakeTimeInvariantC3MIQP(cartpole, Q, R, G, U, x_des, QN, options, N)

#print(Qp)

x0 = np.zeros((4,1))

x0[0] = 0.1
x0[2] = 0.3

delta_add = np.zeros((n+m+k, 1))
w_add = np.zeros((n+m+k, 1))

#input = [5]

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
# plt.plot(time_x, x.T)
# plt.show()

