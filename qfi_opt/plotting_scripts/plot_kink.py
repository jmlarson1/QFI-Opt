#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from qfi_opt import spin_models

N = 4
dissipation = 1
G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

np.random.seed(6)
x0 = np.random.uniform(0, 1, 4)
x1 = np.random.uniform(0, 1, 4)

disp = 1 / 50
vals = np.arange(0, 1 + disp, disp)

results = {}
for i, alpha in enumerate(vals):
    params = alpha * x0 + (1 - alpha) * x1

    rho = spin_models.simulate_OAT(params, N, dissipation_rates=dissipation)

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
