import numpy as np
from pyc3 import LCS, C3MIQP, C3Options

def C3(x0, opt, n, m, k, N, delta_option):

    #Initialize delta and w variables (in R^{n+m+k}) elements correspond to [ state, contact force, input ]
    delta_add = np.zeros((n+m+k, 1))
    w_add = np.zeros((n+m+k, 1))

    if (delta_option == 1):
        delta_add[0:n] = x0

    delta = []
    w = []

    for j in range(N):
        delta.append(delta_add)
        w.append(w_add)

    input = opt.Solve(x0, delta, w)

    return input