import numpy as np
from scipy import linalg

def system_costs(A, B):

    n=4
    m=2
    k=1
    N=10
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
    QN = linalg.solve_discrete_are(A, B, Q, R)

    x0 = np.zeros((4,1))
    x0[0] = 0.1
    x0[2] = 0.3

    return Q,R,G,U,x_des,QN, x0, N