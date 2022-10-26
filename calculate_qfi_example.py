#!/usr/bin/env python3
import itertools
import numpy as np
import run_OAT


def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2


np.set_printoptions(linewidth=200)


def compute_QFI(rho: np.ndarray, G: np.ndarray, tol: float = 1e-6) -> float:
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    num_vals = len(eigvals)

    # Compute QFI
    running_sum = 0
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if np.isclose(denom, 0, atol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                running_sum += numer / denom * np.linalg.norm(term) ** 2

    return 4 * running_sum


N = 4
noise = 0
G = run_OAT.collective_op(run_OAT.pauli_Z, N) / (2 * N)

# Let's try calculating the QFI at all corner points of the domain:
all_perms = [",".join(seq) for seq in itertools.product("01", repeat=4)]
for perm in all_perms:
    params = np.fromstring(perm, dtype=int, sep=",")
    rho = run_OAT.simulate_OAT(N, params, noise)
    qfi = compute_QFI(rho, G)
    print(f"QFI is {qfi} for {params}")

# Let's try calculating the QFI at some random points in the domain:
np.random.seed(0)
for _ in range(10):
    params = np.random.uniform(0, 1, 4)
    rho = run_OAT.simulate_OAT(N, params, noise)
    qfi = compute_QFI(rho, G)
    print(f"QFI is {qfi} for {params}")
