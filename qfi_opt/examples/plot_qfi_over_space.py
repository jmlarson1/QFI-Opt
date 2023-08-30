#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_eigendecompotion, compute_QFI

num_spins = 4
dissipation = 0
op = spin_models.collective_op(spin_models.PAULI_Z, num_spins) / (2 * num_spins)

num_pts = 11
x_ = np.linspace(0.0, 1.0, num_pts)
# x, y, z = np.meshgrid(x_, x_, x_, indexing="ij")
y, z = np.meshgrid(x_, x_, indexing="ij")
obj_vals = np.zeros_like(y)

for i in range(num_pts):
    print(i, flush=True)
    for j in range(num_pts):
        params = np.array([0.5, y[i, j], z[i, j], 0])
        rho = spin_models.simulate_OAT(params, num_spins, dissipation_rates=dissipation)
        vals, vecs = compute_eigendecompotion(rho)
        qfi = compute_QFI(vals, vecs, op)
        obj_vals[i, j] = qfi

fig, ax = plt.subplots()

CS = ax.contour(y, z, obj_vals)
cbar = plt.colorbar(CS)
cbar.set_label("QFI")

plt.title(f"max QFI = {np.max(obj_vals)}")

plt.savefig("contours_with_first_param_half_last_param_zero_" + str(num_pts) + ".png", dpi=300)
