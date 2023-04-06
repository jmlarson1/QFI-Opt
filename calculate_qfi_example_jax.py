#!/usr/bin/env python3
import itertools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import run_OAT_jax as run_OAT

COMPLEX_DTYPE = jnp.complex128


def variance(rho: jnp.ndarray, G: jnp.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2


def compute_QFI(rho: jnp.ndarray, G: jnp.ndarray, tol: float = 1e-8) -> float:
    # Compute eigendecomposition for rho
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    num_vals = len(eigvals)

    # Compute QFI
    running_sum = 0

    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not jnp.isclose(denom, 0, atol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                running_sum += numer / denom * jnp.linalg.norm(term) ** 2

    return 4 * running_sum


def compute_eigh(rho: jnp.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = jnp.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]

    return eigvals, eigvecs


def compute_running_sum(eigvals, eigvecs, G: jnp.ndarray, tol: float = 1e-8) -> float:
    num_vals = len(eigvals)

    # Compute QFI
    running_sum = 0

    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not jnp.isclose(denom, 0, atol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                running_sum += numer / denom * jnp.linalg.norm(term) ** 2

    return 4 * running_sum


if __name__ == "__main__":
    N = 4
    dissipation = 0.0
    G = run_OAT.collective_op(run_OAT.pauli_Z, N) / (2 * N)
    # Let's try calculating the QFI at all corner points of the domain:
    all_perms = [",".join(seq) for seq in itertools.product("01", repeat=4)]
    for perm in all_perms:
        params = jnp.fromstring(perm, dtype=int, sep=",")
        rho = run_OAT.simulate_OAT(params, N, dissipation)
        qfi = compute_QFI(rho, G)
        print(f"QFI is {qfi} for {params}")
    # Let's try calculating the QFI at some random points in the domain:

    np.random.seed(0)
    for _ in range(10):
        params_f = np.random.uniform(0, 1, 4)
        params = jnp.array(params_f, dtype=COMPLEX_DTYPE)
        rho = run_OAT.simulate_OAT(params, N, dissipation)
        qfi = compute_QFI(rho, G)
        print(f"QFI is {qfi} for {params}")
