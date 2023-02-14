#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

import run_OAT
from calculate_qfi_example import compute_QFI

N = 4
noise = 1
G = run_OAT.collective_op(run_OAT.pauli_Z, N) / (2 * N)

np.random.seed(6)
x0 = np.random.uniform(0, 1, 4)
x1 = np.random.uniform(0, 1, 4)

disp = 1 / 50
vals = np.arange(0, 1 + disp, disp)

results = {}
for i, alpha in enumerate(vals):
    params = alpha * x0 + (1 - alpha) * x1

    rho = run_OAT.simulate_OAT(N, params, noise)

    u, v = np.linalg.eig(rho)
    results[i] = [u, v]

Y = np.array([np.sort(np.real(results[i][0])) for i in range(len(results))])

plt.plot(Y, linewidth=6, alpha=0.8, solid_joinstyle="miter")
plt.xticks([0, len(vals)], ["$x_0$", "$x_1$"])
plt.xlabel("Parameters $x$")
plt.ylabel("Eigenvalues of $\\rho(x)$")
# plt.savefig("Initial.png", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig("kink.png", dpi=300, bbox_inches="tight")
plt.close()
