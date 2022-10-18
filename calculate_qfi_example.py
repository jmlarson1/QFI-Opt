#!/usr/bin/env python3
import itertools
import numpy as np
import run_OAT


def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real**2


def compute_QFI(rho: np.ndarray, G: np.ndarray, tol: float = 1e-8) -> float:
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]

    n0 = len(eigvals)

    # Extract nonzero eigenvalues (and corresponding eigenvectors)
    zero_inds = np.isclose(eigvals, np.zeros(n0), rtol=tol, atol=tol)
    nonzero_inds = np.logical_not(zero_inds)
    n1 = sum(nonzero_inds)
    nonzero_eigvals = eigvals[nonzero_inds]
    nonzero_eigvecs = eigvecs[nonzero_inds]

    # Compute QFI
    running_sum = 0
    for i in range(n1):
        for j in range(i+1, n1):
            denom = nonzero_eigvals[i] + nonzero_eigvals[j]
            if denom > tol:
                numer = (nonzero_eigvals[i] - nonzero_eigvals[j])**2
                term = nonzero_eigvecs[i].conj() @ G @ nonzero_eigvecs[j]
                running_sum += (numer/denom)*np.linalg.norm(term)**2

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
    rho = run_OAT.simulate_OAT(N, params, 0)
    qfi = compute_QFI(rho, G)
    print(f"QFI is {qfi} for {params}")
