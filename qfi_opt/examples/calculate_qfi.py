#!/usr/bin/env python3
import numpy as np

from qfi_opt import spin_models

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2


def compute_eigendecompotion(rho: np.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    return eigvals, eigvecs


def compute_QFI(eigvals: jnp.ndarray, eigvecs: jnp.ndarray, G: jnp.ndarray, tol: float = 1e-8) -> jnp.ndarray:
    # Note: The eigenvectors must be rows of eigvecs
    num_vals = len(eigvals)

    # Compute QFI
    running_sum = jnp.array(0)
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not jnp.isclose(denom, 0, atol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                running_sum += numer / denom * jnp.linalg.norm(term) ** 2

    return 4 * running_sum

jacrev_compute_QFI = jax.jacrev(compute_QFI, argnums=(0,1))

if __name__ == "__main__":
    num_spins = 4
    dissipation = 0
    op = spin_models.collective_op(spin_models.PAULI_Z, num_spins) / (2 * num_spins)

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

            params = 0.5 * np.ones(num_params)
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            vals, vecs = compute_eigendecompotion(rho)
            qfi = compute_QFI(vals, vecs, op)
            print(f"QFI is {qfi} for {params}")
            jac = jacrev_compute_QFI(vals, vecs, op)
            print(f"Jacobian is {jac}")

            params[-1] = 0.0
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            vals, vecs = compute_eigendecompotion(rho)
            qfi = compute_QFI(vals, vecs, op)
            print(f"QFI is {qfi} for {params}")
            jac = jacrev_compute_QFI(vals, vecs, op)
            print(f"Jacobian is {jac}")

            params[-1] = 1.0
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            vals, vecs = compute_eigendecompotion(rho)
            qfi = compute_QFI(vals, vecs, op)
            print(f"QFI is {qfi} for {params}")
            jac = jacrev_compute_QFI(vals, vecs, op)
            print(f"Jacobian is {jac}")
