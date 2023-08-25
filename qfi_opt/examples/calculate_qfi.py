#!/usr/bin/env python3
import numpy as np

from qfi_opt import spin_models


def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2

def compute_eigendecompotion(rho: np.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    return eigvals, eigvecs


def compute_QFI(eigvals: np.ndarray, eigvecs: np.ndarray, G: np.ndarray, tol: float = 1e-8) -> float:
    # Note: The eigenvectors must be rows of eigvecs

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

    num_rand_pts = 2
    # Calculate QFI for models at random points in the domain.
    for num_params in [4, 5]:
        match num_params:
            case 4:
                models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
            case 5:
                models = ["simulate_TAT", "simulate_local_TAT_chain"]

        for model in models:
            print(model)
            np.random.seed(0)
            obj = getattr(spin_models, model)
            # for _ in range(num_rand_pts):
            #     params = np.random.uniform(0, 1, num_params)
            #     rho = obj(params, N, dissipation_rates=dissipation)
            #     qfi = compute_QFI(compute_eigendecompotion(rho), G)
            #     print(f"QFI is {qfi} for {params}")

            params = 0.5 * np.ones(num_params)
            rho = obj(params, N, dissipation_rates=dissipation)
            V,E = compute_eigendecompotion(rho)
            qfi = compute_QFI(V,E, G)
            print(f"QFI is {qfi} for {params}")

            params[-1] = 0.0
            rho = obj(params, N, dissipation_rates=dissipation)
            V,E = compute_eigendecompotion(rho)
            qfi = compute_QFI(V,E, G)
            print(f"QFI is {qfi} for {params}")

            params[-1] = 1.0
            rho = obj(params, N, dissipation_rates=dissipation)
            V,E = compute_eigendecompotion(rho)
            qfi = compute_QFI(V,E, G)
            print(f"QFI is {qfi} for {params}")
