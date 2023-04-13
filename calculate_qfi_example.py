#!/usr/bin/env python3
import itertools

import numpy as np

import spin_models


def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2


def compute_QFI(rho: np.ndarray, G: np.ndarray, tol: float = 1e-8) -> float:
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    num_vals = len(eigvals)

    # Compute QFI
    running_sum = 0

    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not np.isclose(denom, 0, atol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                running_sum += numer / denom * np.linalg.norm(term) ** 2

    return 4 * running_sum


if __name__ == "__main__":
    N = 4
    dissipation = 0
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    # # Let's try calculating the QFI at all corner points of the domain:
    # all_perms = [",".join(seq) for seq in itertools.product("01", repeat=4)]
    # for perm in all_perms:
    #     params = np.fromstring(perm, dtype=int, sep=",")
    #     rho = spin_models.simulate_OAT(params, N, dissipation)
    #     qfi = compute_QFI(rho, G)
    #     print(f"QFI is {qfi} for {params}")

    num_rand_pts = 2
    # Calculate QFI for models at random points in the domain.
    for num_params in [4, 5]:
        match num_params:
            case 4:
                models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
            case 5:
                models = ["simulate_TAT", "simulate_local_TAT_chain"]

        for model in models:
            np.random.seed(0)
            obj = getattr(spin_models, model)
            for _ in range(num_rand_pts):
                params = np.random.uniform(0, 1, num_params)
                rho = obj(params, N, dissipation)
                qfi = compute_QFI(rho, G)
                print(f"QFI is {qfi} for {params}")
