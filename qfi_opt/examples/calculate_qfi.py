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
        #eigenvector_bases = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

        for k in range(num_params):
            # compute gradients of each eigenvalue
            psi_grad_k, lambda_grad_k = get_matrix_grads_hail_mary(rho, dA[k], eigvals, eigvecs, tol)
            #psi_grad_k, lambda_grad_k, basis_k = get_matrix_grads_rotate(rho, dA[k], eigvals, eigvecs, tol)
            psi_grads[k] = psi_grad_k
            lambda_grads[k] = lambda_grad_k
            #eigenvector_bases[k] = basis_k

    # NOW COMPUTE
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            diff = eigvals[i] - eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                #f_quotient, g_quotient = qfi_quotient2(eigvals[i], eigvals[j], eigvecs[i], eigvecs[j], dA)
                f_quotient, g_quotient = qfi_quotient3(eigvals[i], eigvals[j], lambda_grads[:, [i, j]])
                f_modulus, g_modulus = qfi_modulus(G, psi_grads, i, j, eigvecs[i], eigvecs[j])
                #f_modulus, g_modulus = qfi_modulus2(G, psi_grads, i, j, eigenvector_bases)
                running_sum += f_quotient * f_modulus
                if grad.size > 0:
                    grad[:] += f_quotient * g_modulus + f_modulus * g_quotient

    if grad.size > 0:
        return 4 * running_sum, 4 * grad
    else:
        return 4 * running_sum, []


def get_matrix_grads_sylvester(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")

    # force Hermitianness:
    dA = (dA + dA.conj().T) / 2.0

    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2], eigvals[ind1], atol=tol):
                    break  # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            group_set = np.arange(ind1, ind2)

            if group_set.size == 0:
                group_set = [ind2]

            # Two cases - either the eigenvalue has multiplicity one or it doesn't.
            if len(group_set) > 1:
                not_in_group_set = np.setdiff1d(np.arange(dim), group_set)
                A_group = np.diag(eigvals[group_set])
                A_not_in_group = np.diag(eigvals[not_in_group_set])
                rotation = eigvecs[group_set].conj() @ dA @ eigvecs[not_in_group_set].T
                sol = solve_sylvester(A_group, A_not_in_group, rotation)
                psi_grads[group_set] = sol @ eigvecs[not_in_group_set].conj()

            else: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                M = np.hstack((rho - eigvals[ind1] * np.eye(dim), -np.expand_dims(eigvecs[ind1].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[ind1].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[ind1].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                psi_grads[ind1] = np.squeeze(sol[:dim])

    return psi_grads

def get_matrix_grads_naive(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    # force Hermitianness:
    dA = (dA + dA.conj().T) / 2.0

    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2], eigvals[ind1], atol=tol):
                    break  # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            group_set = np.arange(ind1, ind2)

            if group_set.size == 0:
                group_set = [ind2]

            # Two cases - either the eigenvalue has multiplicity one or it doesn't.
            if len(group_set) > 1:
                lhs = rho - eigvals[group_set[-1]] * np.eye(dim)
                lhs = np.hstack((lhs, -eigvecs[group_set].T))
                lhs_row2 = np.hstack((-eigvecs[group_set].conj(), np.zeros((len(group_set), len(group_set)))))
                lhs = np.vstack((lhs, lhs_row2))
                #Lambda_prime = eigvecs[group_set].conj() @ dA @ eigvecs[group_set].T
                rhs = -dA @ eigvecs[group_set].T# + eigvecs[group_set].T @ Lambda_prime
                rhs = np.vstack((rhs, np.zeros((len(group_set), len(group_set)))))
                sol = np.linalg.solve(lhs, rhs)
                psi_grads[group_set] = sol[:dim, :].T
                Lambda_prime = sol[dim:, :]
                lambda_grads[group_set] = np.real(np.diag(Lambda_prime))
            else: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                M = np.hstack((rho - eigvals[ind1] * np.eye(dim), -np.expand_dims(eigvecs[ind1].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[ind1].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[ind1].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                psi_grads[ind1] = np.squeeze(sol[:dim])
                lambda_grads[ind1] = np.real(sol[dim])

    return psi_grads, lambda_grads


def get_matrix_grads_hail_mary(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    # force Hermitianness:
    dA = (dA + dA.conj().T) / 2.0

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
            # If the eigenvalue has multiplicity, we're going to perturb the matrix to avoid instabilities
            for j in range(1, len(group_set)):
                # rank one perturbation
                rho += j * tol * eigvecs[group_set[j]].conj() @ eigvecs[group_set[j]].T
            for j in group_set: # The eigenvalue has multiplicity one and we can do the more obvious thing:
                M = np.hstack((rho - eigvals[j] * np.eye(dim), -np.expand_dims(eigvecs[j].T, 1)))
                M = np.vstack((M, np.expand_dims(np.hstack((eigvecs[j].conj(), 0)), 0)))
                rhs = np.vstack((np.expand_dims(-dA @ eigvecs[j].T, 1), 0))
                sol = np.linalg.solve(M, rhs)
                psi_grads[j] = np.squeeze(sol[:dim])
                lambda_grads[j] = np.real(sol[dim])

    return psi_grads, lambda_grads


def get_matrix_grads_rotate(rho, dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    psi_grads = np.zeros((dim, dim), dtype="cdouble")
    lambda_grads = np.zeros(dim)

    # force Hermitianness:
    dA = (dA + dA.conj().T) / 2.0

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
                Lambda_prime = sol[dim:, :] # this should, in infinite precision, be equal to H. 
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


def qfi_quotient(lambda_i, lambda_j, psi_i, psi_j, dA):

    dim = np.shape(dA)[0]

    diff = lambda_i - lambda_j
    sum = lambda_i + lambda_j

    f = diff ** 2 / sum

    # compute a single subgradient in the nonsmooth case
    fprimeg = 2 * diff * sum * (np.outer(psi_i, psi_i.conj()) - np.outer(psi_j, psi_j.conj()))
    gprimef = (diff ** 2) * (np.outer(psi_i, psi_i.conj()) + np.outer(psi_j, psi_j.conj()))
    drhoQ = (fprimeg - gprimef) / (sum ** 2)

    g = np.zeros(dim)
    # trace inner product
    for k in range(dim):
        der = (dA[k].T.ravel()).conj().T @ (drhoQ.T).ravel()
        g[k] = np.real(der.tolist())

    g = np.real(g)

    return f, g


def qfi_quotient2(lambda_i, lambda_j, psi_i, psi_j, dA):

    dim = np.shape(dA)[0]

    diff = lambda_i - lambda_j
    sum = lambda_i + lambda_j

    f = diff ** 2 / sum

    g = np.zeros(dim)
    for k in range(dim):
        dk_lambda_i = psi_i.conj() @ dA[k] @ psi_i.T
        dk_lambda_j = psi_j.conj() @ dA[k] @ psi_j.T

        g[k] = np.real((2 * diff * sum * (dk_lambda_i - dk_lambda_j) - (dk_lambda_i + dk_lambda_j) * diff ** 2) / (sum ** 2))

    return f, g

def qfi_quotient3(lambda_i, lambda_j, lambda_grads):

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


def qfi_modulus(G, psi_grads, i, j, psi_i, psi_j):

    dim = np.shape(psi_grads)[0]
    g = np.zeros(dim)

    ip = psi_i.conj() @ G @ psi_j.T

    f = np.absolute(ip) ** 2

    for k in range(dim):
        d_xk_psi_i = psi_grads[k, i]
        d_xk_psi_j = psi_grads[k, j]
        der_product = d_xk_psi_i.conj() @ G @ psi_j.T + psi_i.conj() @ G @ d_xk_psi_j.T
        g[k] = 2 * np.real(ip) * np.real(der_product) + 2 * np.imag(ip) * np.imag(der_product)

    return f, g


def qfi_modulus2(G, psi_grads, i, j, eigenvectors):

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

    ## HIGHER DIM EXAMPLE
    dissipation = 0.01
    model = 'local_TAT'
    N = 4
    coupling_exponent = 0
    layers = 5
    num_params = 2 * layers + 3

    obj = getattr(sm, f'simulate_{model}_chain')
    obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N) / (2 * N), 'N': N, 'dissipation': dissipation,
                  'coupling_exponent': coupling_exponent}

    # LOWER DIM EXAMPLE
    #dissipation = 0.01
    #num_params = 4
    #model = "simulate_XX_chain"
    #N = 5
    #obj = getattr(sm, model)
    #obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N) / (2 * N), 'N': N, 'dissipation': dissipation}

    get_jacobian = sm.get_jacobian_func(obj)

    rho = obj(params, N, dissipation_rates=dissipation)
    vals, vecs = compute_eigendecomposition(rho)
    qfi_grad = np.zeros(num_params)
    qfi = compute_QFI(rho, vals, vecs, params, obj_params=obj_params, grad=qfi_grad, get_jacobian=get_jacobian)

    #grad_of_rho = get_jacobian(params, N, dissipation_rates=dissipation)

