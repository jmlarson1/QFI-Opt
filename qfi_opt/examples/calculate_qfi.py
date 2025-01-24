#!/usr/bin/env python3
import numpy as np
from scipy.io import savemat, loadmat
from scipy.linalg import solve_sylvester

import ipdb

from qfi_opt import spin_models as sm

def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2


def compute_eigendecomposition(rho: np.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvecs = eigvecs.T  # make the k-th eigenvector eigvecs[k, :] = eigvecs[k]
    # consistent sorting:
    eigvals = np.real(eigvals)
    sort_inds = np.argsort(eigvals)
    eigvals = eigvals[sort_inds]
    eigvecs = eigvecs[sort_inds]
    return eigvals, eigvecs


def compute_QFI(rho: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray, params: np.ndarray, obj_params, tol: float = 1e-8, etol_scale: float = 10, grad=np.empty(0), get_jacobian=[]):
    # Note: The eigenvectors must be rows of eigvecs
    num_vals = len(eigvals)
    num_params = len(params)

    G = obj_params["G"]

    # There should never be negative eigenvalues, so their magnitude gives an
    # empirical estimate of the numerical accuracy of the eigendecomposition.
    # We discard any QFI terms denominators within an order of magnitude of
    # this value.
    tol = max(tol, -etol_scale * np.min(eigvals))

    # Compute QFI and grad
    running_sum = 0

    if grad.size > 0:

        dA = get_jacobian(params, obj_params["N"], dissipation_rates=obj_params["dissipation"])
        dA = np.transpose(dA, (2, 0, 1))

        grad[:] = np.zeros(num_params)
        psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")
        lambda_grads = np.zeros((num_params, num_vals))
        eigenvector_bases = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

        for k in range(num_params):
            # compute gradients of each eigenvalue
            psi_grad_k, lambda_grad_k, basis_k = get_matrix_grads_rotate(rho, dA[k], eigvals, eigvecs, tol)
            psi_grads[k] = psi_grad_k
            lambda_grads[k] = lambda_grad_k
            eigenvector_bases[k] = basis_k

    # NOW COMPUTE
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            diff = eigvals[i] - eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                f_quotient, g_quotient = qfi_quotient(eigvals[i], eigvals[j], lambda_grads[:, [i, j]])
                f_modulus, g_modulus = qfi_modulus(G, psi_grads, i, j, eigenvector_bases)
                running_sum += f_quotient * f_modulus
                if grad.size > 0:
                    grad[:] += f_quotient * g_modulus + f_modulus * g_quotient

    if grad.size > 0:
        return 4 * running_sum, 4 * grad
    else:
        return 4 * running_sum, []

def check_close_entries(arr, tol):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diff = arr[j] - arr[i]
            if np.isclose(diff, 0, atol=tol, rtol=tol):
                return True
    return False

def compute_QFI2(rho: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray, params: np.ndarray, obj_params,
                    tol: float = 1e-8, etol_scale: float = 10, grad=np.empty(0), get_jacobian=[], get_hessian=[]):

    # Note: The eigenvectors must be rows of eigvecs
    num_vals = len(eigvals)
    num_params = len(params)

    G = obj_params["G"]

    # There should never be negative eigenvalues, so their magnitude gives an
    # empirical estimate of the numerical accuracy of the eigendecomposition.
    # We discard any QFI terms denominators within an order of magnitude of
    # this value.
    tol = max(tol, -etol_scale * np.min(eigvals))

    # Compute QFI and grad
    running_sum = 0

    if grad.size > 0:

        dA = get_jacobian(params, obj_params["N"], dissipation_rates=obj_params["dissipation"])
        dA = np.transpose(dA, (2, 0, 1))

        # do we have any repeating eigenvalues?
        if check_close_entries(eigvals, tol):
            print("Some eigenvalues are sufficiently close, need to compute diagonal Hessian. eigvals: ", eigvals)
            d2A = get_hessian(params, obj_params["N"], dissipation_rates=obj_params["dissipation"])
            d2A = np.transpose(d2A, (2, 0, 1))
        else:
            d2A = np.empty(num_params)

        grad[:] = np.zeros(num_params)
        psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")
        lambda_grads = np.zeros((num_params, num_vals))
        eigenvector_bases = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

        for k in range(num_params):
            # compute gradients of each eigenvalue
            # I am only passing k as the last argument right now for the sake of debugging which partials are screwy.
            psi_grad_k, lambda_grad_k, basis_k = get_matrix_grads_hessian(rho, dA[k], d2A[k], eigvals, eigvecs, tol, k)
            psi_grads[k] = psi_grad_k
            lambda_grads[k] = lambda_grad_k
            eigenvector_bases[k] = basis_k

    # NOW COMPUTE
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            diff = eigvals[i] - eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff**2 / denom, 0, atol=tol, rtol=tol):
                # can i possibly help the scaling here to avoid numerical blowups?
                # idea: try to get the sum (denom) to be approximately 1 so the division by denom(**2) doesn't hurt too bad.
                scaling_factor = 1.0 #denom
                eigvals_i = eigvals[i] / scaling_factor
                eigvals_j = eigvals[j] / scaling_factor
                lambda_grads_ij = lambda_grads[:, [i, j]] / scaling_factor
                f_quotient, g_quotient = qfi_quotient(eigvals_i, eigvals_j, lambda_grads_ij)
                f_modulus, g_modulus = qfi_modulus(G, psi_grads, i, j, eigenvector_bases)
                # scale back (multiply f_quotient and g_quotient by an extra scaling_factor)
                running_sum += f_quotient * scaling_factor * f_modulus
                if grad.size > 0:
                    grad[:] += scaling_factor * (f_quotient * g_modulus + f_modulus * g_quotient)

    if grad.size > 0:
        return 4 * running_sum, 4 * grad
    else:
        return 4 * running_sum, []


def get_matrix_grads_rotate(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2], eigvals[ind1], atol=tol, rtol=tol):
                    break  # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            group_set = np.arange(ind1, ind2)

            if group_set.size == 0:
                group_set = [ind2]

            # Two cases - either the eigenvalue has multiplicity one or it doesn't.
            if len(group_set) > 1:
                Lambda_prime = eigvecs[group_set].conj() @ dA @ eigvecs[group_set].T
                H = Lambda_prime @ Lambda_prime
                eigvalsH, eigvecsH = np.linalg.eigh(H)
                rotated_eigvecs = eigvecs[group_set].T @ eigvecsH
                lhs = rho - eigvals[group_set[-1]] * np.eye(dim)
                lhs = np.hstack((lhs, -rotated_eigvecs))
                lhs_row2 = np.hstack((-rotated_eigvecs.T.conj(), np.zeros((len(group_set), len(group_set)))))
                lhs = np.vstack((lhs, lhs_row2))
                rhs = -dA @ rotated_eigvecs
                rhs = np.vstack((rhs, np.zeros((len(group_set), len(group_set)))))
                sol = np.linalg.solve(lhs, rhs)
                psi_grads[group_set] = sol[:dim, :].T
                Lambda_prime = sol[dim:, :] 
                lambda_grads[group_set] = np.real(np.diag(Lambda_prime))

                # key: let the routine that called this subroutine know we rotated the eigvecs
                eigvecs[group_set] = rotated_eigvecs.T
            else: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                M = np.hstack((rho - eigvals[ind1] * np.eye(dim), -np.expand_dims(eigvecs[ind1].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[ind1].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[ind1].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                psi_grads[ind1] = np.squeeze(sol[:dim])
                lambda_grads[ind1] = np.real(sol[dim])

    return psi_grads, lambda_grads, eigvecs


def get_matrix_grads_hessian(rho, dA, d2A, eigvals, eigvecs, tol, k):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2] - eigvals[ind1], 0, atol=tol, rtol=tol):
                    break  # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            group_set = np.arange(ind1, ind2)

            if group_set.size == 0:
                group_set = [ind2]

            if len(group_set) > 1:
                M1 = eigvecs[group_set].conj() @ dA @ eigvecs[group_set].T
                eigvals_sub, eigvecs_sub = compute_eigendecomposition(M1)
                eigvals_sub = np.real(eigvals_sub)
                lambda_grads[group_set] = eigvals_sub
                if check_close_entries(eigvals_sub, tol):
                    print("Derivative eigenvalues are numerically close. group_set: ", group_set, "lambda_grads: ", eigvals_sub, "partial: ", k)
                rotated_eigvecs = eigvecs[group_set].T @ eigvecs_sub.T
                V = np.linalg.solve(rho - eigvals[ind1] * np.eye(dim),
                                    -1.0 * dA @ eigvecs[group_set].T + rotated_eigvecs @ np.diag(eigvals_sub) @ eigvecs_sub.conj())
                M2 = rotated_eigvecs.T.conj() @ (d2A @ rotated_eigvecs - 2.0 * V @ eigvecs_sub.T @ np.diag(
                    eigvals_sub) + 2 * dA @ V @ eigvecs_sub.T)
                C = np.diag(np.diag(-1.0 * rotated_eigvecs.T.conj() @ V @ eigvecs_sub.T))
                for i in range(len(group_set)):
                    for j in range(len(group_set)):
                        if i != j:
                            C[i, j] = M2[i, j] / (2 * (eigvals_sub[j] - eigvals_sub[i]))

                psi_grads[group_set] = (V @ eigvecs_sub.T + rotated_eigvecs @ C).T

                # key: let the routine that called this subroutine know we rotated the eigvecs
                eigvecs[group_set] = rotated_eigvecs.T

            else: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                # key observation: d2A appears nowhere here!
                M = np.hstack((rho - eigvals[ind1] * np.eye(dim), -np.expand_dims(eigvecs[ind1].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[ind1].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[ind1].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                psi_grads[ind1] = np.squeeze(sol[:dim])
                lambda_grads[ind1] = np.real(sol[dim])

    return psi_grads, lambda_grads, eigvecs


def qfi_quotient(lambda_i, lambda_j, lambda_grads):

    dim = np.shape(lambda_grads)[0]

    diff = lambda_i - lambda_j
    sum = lambda_i + lambda_j

    f = diff ** 2 / sum

    g = np.zeros(dim)
    for k in range(dim):
        dk_lambda_i = lambda_grads[k, 0]
        dk_lambda_j = lambda_grads[k, 1]

        g[k] = np.real((2 * diff * sum * (dk_lambda_i - dk_lambda_j) - (dk_lambda_i + dk_lambda_j) * diff ** 2) / (sum ** 2))

    return f, g


def qfi_modulus(G, psi_grads, i, j, eigenvectors):

    dim = np.shape(psi_grads)[0]
    g = np.zeros(dim)

    # WLOG:
    psi_i = eigenvectors[0, i]
    psi_j = eigenvectors[0, j]
    ip = psi_i.conj() @ G @ psi_j.T

    f = np.absolute(ip) ** 2

    for k in range(dim):
        d_xk_psi_i = psi_grads[k, i]
        d_xk_psi_j = psi_grads[k, j]
        psi_i = eigenvectors[k, i]
        psi_j = eigenvectors[k, j]
        der_product = d_xk_psi_i.conj() @ G @ psi_j.T + psi_i.conj() @ G @ d_xk_psi_j.T
        g[k] = 2 * np.real(ip) * np.real(der_product) + 2 * np.imag(ip) * np.imag(der_product)

    return f, g


if __name__ == "__main__":

    # read in a point to evaluate from params.mat
    loaded_file = loadmat('params.mat')
    params = loaded_file['params'][0]

    dissipation = 0.01
    model = 'local_TAT'
    N = 4
    coupling_exponent = 0
    ## HIGHER DIM EXAMPLE
    # layers = 5
    ## LOWER DIM EXAMPLE
    layers = 1
    num_params = 2 * layers + 3

    obj = getattr(sm, f'simulate_{model}_chain')
    obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N) / (2 * N), 'N': N, 'dissipation': dissipation,
                  'coupling_exponent': coupling_exponent}

    get_jacobian = sm.get_jacobian_func(obj)
    get_hessian = sm.get_hessian_diag_func(obj)

    rho = obj(params, N, dissipation_rates=dissipation)
    vals, vecs = compute_eigendecomposition(rho)
    qfi_grad = np.zeros(num_params)
    ## DOESN'T USE HESSIAN
    #qfi = compute_QFI(rho, vals, vecs, params, obj_params=obj_params, grad=qfi_grad, get_jacobian=get_jacobian)
    ## USES HESSIAN ONLY WHEN EIGENVALUES ARE CLOSE TO IDENTIFY INVARIANT SUBSPACE
    qfi = compute_QFI2(rho, vals, vecs, params, obj_params=obj_params, grad=qfi_grad, get_jacobian=get_jacobian, get_hessian=get_hessian)


