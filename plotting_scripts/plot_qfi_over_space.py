#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

import spin_models
from calculate_qfi_example import compute_QFI

N = 4
dissipation = 0
G = spin_models.collective_op(spin_models.pauli_Z, N) / (2 * N)

num_pts = 11
x_ = np.linspace(0.0, 1.0, num_pts)
# x, y, z = np.meshgrid(x_, x_, x_, indexing="ij")
y, z = np.meshgrid(x_, x_, indexing="ij")
obj_vals = np.zeros_like(y)

for i in range(num_pts):
    print(i, flush=True)
    for j in range(num_pts):
        #     for k in range(num_pts):
        #         params = np.array([x[i,j,k], y[i,j,k], z[i,j,k], 0])
        #         rho = spin_models.simulate_OAT(N, params, dissipation)
        #         qfi = compute_QFI(rho, G)
        params = np.array([0.5, y[i, j], z[i, j], 0])
        rho = spin_models.simulate_OAT(N, params, dissipation)
        qfi = compute_QFI(rho, G)
        obj_vals[i, j] = qfi

fig, ax = plt.subplots()

CS = ax.contour(y, z, obj_vals)
cbar = plt.colorbar(CS)
cbar.set_label("QFI")

plt.title(f"max QFI = {np.max(obj_vals)}")

plt.savefig("contours_with_first_param_half_last_param_zero_" + str(num_pts) + ".png", dpi=300)
