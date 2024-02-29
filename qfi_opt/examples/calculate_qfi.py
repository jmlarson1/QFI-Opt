#!/usr/bin/env python3
import numpy as np
import ipdb
from scipy.io import savemat
from scipy.linalg import solve_sylvester

import sys

sys.path.append('../../')

from qfi_opt import spin_models


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


def compute_QFI(eigvals: np.ndarray, eigvecs: np.ndarray, G: np.ndarray, A: np.ndarray, dA: np.ndarray, d2A: np.ndarray, grad,
                tol: float = 1e-8, etol_scale: float = 10) -> float:
    # Note: The eigenvectors must be rows of eigvecs
    num_vals = len(eigvals)
    num_params = dA.shape[0]

    # There should never be negative eigenvalues, so their magnitude gives an
    # empirical estimate of the numerical accuracy of the eigendecomposition.
    # We discard any QFI terms denominators within an order of magnitude of
    # this value.
    tol = max(tol, -etol_scale * np.min(eigvals))

    # Compute QFI and grad
    running_sum = 0

    if grad.size > 0:

        grad[:] = np.zeros(num_params)
        # compute gradients of each eigenvalue
        #lambda_grads, psi_grads, eigvecs = get_matrix_grads(A, dA, d2A, eigvals, eigvecs, tol)
        lambda_grads, psi_grads = get_matrix_grads_lazy(A, dA, eigvals, eigvecs)


    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            diff = eigvals[i] - eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                numer = diff ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                quotient = numer / denom
                squared_modulus = np.absolute(term) ** 2
                running_sum += quotient * squared_modulus
                if grad.size > 0:
                    for k in range(num_params):
                        # fill in gradient
                        grad[k] += kth_partial_derivative(quotient, squared_modulus, eigvals[i], eigvals[j],
                                                        lambda_grads[k, i], lambda_grads[k, j], eigvecs[i], eigvecs[j],
                                                        psi_grads[k, i], psi_grads[k, j], G)

    if grad.size > 0:
        return 4 * running_sum, 4 * grad
    else:
        return 4 * running_sum, []


def compute_QFI_diffrax(eigvals: np.ndarray, eigvecs: np.ndarray, A: np.ndarray, params: np.ndarray, grad,
                get_jacobian, obj_params, tol: float = 1e-8, etol_scale: float = 10) -> float:
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
        lambda_grads = np.zeros((num_params, num_vals))
        psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

        for k in range(num_params):
            # compute gradients of each eigenvalue
            lambda_grad_k, psi_grad_k = get_matrix_grads_sylvester(dA[k], eigvals, eigvecs, tol)
            lambda_grads[k] = lambda_grad_k
            psi_grads[k] = psi_grad_k

    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            diff = eigvals[i] - eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol) and not np.isclose(diff, 0, atol=tol, rtol=tol):
                #ipdb.set_trace()
                numer = diff ** 2
                term = eigvecs[i].conj() @ G @ eigvecs[j]
                quotient = numer / denom
                squared_modulus = np.absolute(term) ** 2
                running_sum += quotient * squared_modulus
                if grad.size > 0:
                    for k in range(num_params):
                        # fill in gradient
                        grad[k] += kth_partial_derivative(quotient, squared_modulus, eigvals[i], eigvals[j],
                                                        lambda_grads[k, i], lambda_grads[k, j], eigvecs[i], eigvecs[j],
                                                        psi_grads[k, i], psi_grads[k, j], G)

    if grad.size > 0:
        return 4 * running_sum, 4 * grad
    else:
        return 4 * running_sum, []


def get_matrix_grads_sylvester(dA, eigvals, eigvecs, tol):

    dim = eigvecs.shape[0]
    lambda_grads = np.zeros(dim, dtype="cdouble")
    psi_grads = np.zeros((dim, dim), dtype="cdouble")

    # force Hermitianness:
    dA = (dA + dA.conj().T) / 2.0


    # group the sorted eigvals by tolerance, intended to help stability of eigenvector derivatives:
    current_ind = 0
    for ind1 in range(dim):
        if current_ind == ind1:
            for ind2 in range(ind1 + 1, dim):
                if not np.isclose(eigvals[ind2], eigvals[ind1], atol=tol, rtol=tol):
                    break # the for loop over ind2
            # we just broke the for loop, so:
            current_ind = ind2

            # do a sylvester solve:
            group_set = np.arange(ind1, ind2)
            # special case when ind2=dim (must be something smarter)
            if group_set.size == 0:
                group_set = [ind2]
            not_in_group_set = np.setdiff1d(np.arange(dim), group_set)
            A_group = np.diag(eigvals[group_set])
            A_not_in_group = np.diag(eigvals[not_in_group_set])
            rotation = eigvecs[group_set].conj() @ dA @ eigvecs[not_in_group_set].T
            sol = solve_sylvester(A_group, -1.0 * A_not_in_group, rotation)
            psi_grads[group_set] = sol @ eigvecs[not_in_group_set].conj()
            #ipdb.set_trace()

            # average eigenvalue:
            multiplicity = len(group_set)
            dLambda = (1.0 / multiplicity) * np.trace(eigvecs[group_set].conj() @ dA @ eigvecs[group_set].T)
            lambda_grads[group_set] = np.ones(multiplicity) * dLambda

    return np.real(lambda_grads), psi_grads.conj()


def get_matrix_grads_lazy(A, dA, eigvals, eigvecs):

    num_vals = len(eigvals)
    num_params = dA.shape[0]
    lambda_grads = np.zeros((num_params, num_vals))
    psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")

    for k in range(num_params):
        for index in range(num_vals):

            dlambda, dpsi = matrixder_lazy(A, dA[k], eigvals[index], eigvecs[index])

            lambda_grads[k, index] = np.real(dlambda)
            psi_grads[k, index] = dpsi.flatten()

    return lambda_grads, psi_grads


def get_matrix_grads(A, dA, d2A, eigvals, eigvecs, tol):

    num_vals = len(eigvals)
    num_params = dA.shape[0]
    lambda_grads = np.zeros((num_params, num_vals))
    psi_grads = np.zeros((num_params, num_vals, num_vals), dtype="cdouble")
    updated_eigvecs = np.zeros((num_vals, num_vals), dtype="cdouble")

    for k in range(num_params):
        already_used = []
        for val in range(num_vals):
            if val not in already_used:
                # check for multiplicity within tolerance
                eigval = eigvals[val]
                indices = np.where(np.abs(eigvals.flatten() - eigval) <= tol)[0]
                already_used = already_used + indices.tolist()

                dlambda_k, dV_k, V_k = matrixder(A, dA[k], d2A[k], eigval, eigvecs[indices])

                lambda_grads[k, indices] = dlambda_k
                psi_grads[k, indices] = dV_k
                updated_eigvecs[indices] = V_k

    return lambda_grads, psi_grads, updated_eigvecs

def matrixder_lazy(A, dA, eigval, eigvec):

    # build the big matrix
    num_vals = A.shape[0]
    Eye = np.eye(num_vals)
    Z = np.zeros((num_vals, num_vals))
    eigvec = eigvec[:, np.newaxis]
    #W = np.concatenate((A - eigval * Eye, Z, -1.0 * eigvec), axis=1, dtype="cdouble")
    W = np.concatenate((A - eigval * Eye, -1.0 * eigvec), axis=1, dtype="cdouble")
    #row = np.append(eigvec.conj().T, eigvec.T)
    row = eigvec.conj().T
    row = np.append(row, 0)
    row = row[:, np.newaxis].T
    #W2 = np.concatenate((Z, A.conj() - eigval * Eye,  -1.0 * eigvec.conj()), axis=1, dtype="cdouble")
    W = np.concatenate((W, row), axis=0, dtype="cdouble")

    # right hand side
    prod = dA @ eigvec
    rhs = -1.0 * prod
    rhs = np.append(rhs, 0)
    #rhs = np.append(rhs, -1.0 * prod.conj())
    rhs = rhs[:, np.newaxis]

    sol = np.linalg.solve(W, rhs)
    ipdb.set_trace()

    # read off values
    dlambda = sol[-1]
    dpsi = sol[:num_vals]

    return dlambda, dpsi


def matrixder(A, dA, ddA, lam, arbV):
    # A is a Hermitian n times n matrix
    # dA is the first derivative of A with respect to a parameter of interest
    # ddA is the second derivative of A with respect to a parameter of interest
    # lam is one eigenvalue of A
    # arbV is an orthonormal basis for the eigenspace associated with lam, of shape n x r (r is multiplicity)

    # because arbV is transposed
    arbV = arbV.T

    # Step 1: Initialize a few structures
    n, r = arbV.shape
    B = np.eye(n)

    # Step 2: set up initial eigenvalue problem
    M = arbV.conj().T @ dA @ arbV
    W = A - lam * B

    # Step 3: (we only want first derivatives, so Step 3 is not necessary)

    # Step 4: Solve the eigenvalue problem
    dLambda, U = np.linalg.eigh(M)
    V = arbV @ U  # this is V in the "correct" basis

    # Step 5: reduced eigenspace
    V1 = np.linalg.solve(W, -dA @ arbV + V @ np.diag(dLambda) @ U.conj().T)

    # Step 6: Compute M2
    if r > 1:  # only do this step if repeated eigenvalue
        M2 = V.conj().T @ ddA @ V + 2.0 * V.conj().T @ (-V1 @ U @ np.diag(dLambda) + dA @ V1 @ U)

    # Step 7: Build C
    D = -1.0 * V.conj().T @ V1 @ U
    C = np.zeros((r, r), dtype="cdouble")
    for i in range(r):
        for j in range(r):
            if i == j:
                C[i, j] = D[i, j]
            else:
                C[i, j] = M2[i, j] / (2 * (dLambda[j] - dLambda[i]))

    # Step 8: Derivatives of eigenvectors
    dV = V1 @ U + V @ C

    return dLambda, dV.T, V.T


def quotient_partial_derivative(lambda_i, lambda_j, d_lambda_i, d_lambda_j):

    squared_diff = (lambda_i - lambda_j) ** 2
    fprimeg = 2 * (lambda_i - lambda_j) * (lambda_i + lambda_j) * (d_lambda_i - d_lambda_j)
    gprimef = squared_diff * (d_lambda_i + d_lambda_j)
    der = (fprimeg - gprimef) / (lambda_i + lambda_j) ** 2

    return der


def modulus_partial_derivative(psi_i, psi_j, d_psi_i, d_psi_j, G):

    inner_product = psi_i.conj() @ G @ psi_j
    left_derivative = d_psi_i.conj() @ G @ psi_j
    right_derivative = psi_i.conj() @ G @ d_psi_j

    real_der = 2 * inner_product.real * (left_derivative.real + right_derivative.real)
    imag_der = 2 * inner_product.imag * (left_derivative.imag + right_derivative.imag)

    der = real_der + imag_der

    return der


def kth_partial_derivative(quotient, modulus, lambda_i, lambda_j, d_lambda_i, d_lambda_j,
                           psi_i, psi_j, d_psi_i, d_psi_j, G):

    quotient_der = quotient_partial_derivative(lambda_i, lambda_j, d_lambda_i, d_lambda_j)
    modulus_der = modulus_partial_derivative(psi_i, psi_j, d_psi_i, d_psi_j, G)

    der = quotient * modulus_der + modulus * quotient_der

    return der


if __name__ == "__main__":
    num_spins = 5
    dissipation = 1.0
    op = spin_models.collective_op(spin_models.PAULI_Z, num_spins) / (2 * num_spins)

    num_rand_pts = 2
    print_precision = 6

    seed = 8888
    np.random.seed(seed)

    for num_params in [4]:#[4, 5]:
        #center = 0.5 * np.ones(num_params)
        center = np.random.uniform(np.zeros(num_params), np.ones(num_params), num_params)
        B = np.eye(num_params)
        h = 1e-6
        match num_params:
            case 4:
                models = ["simulate_OAT"]#, "simulate_ising_chain", "simulate_XX_chain"]
            case 5:
                models = ["simulate_TAT"]#, "simulate_local_TAT_chain"]

        for model in models:
            obj = getattr(spin_models, model)
            get_jacobian = spin_models.get_jacobian_func(obj)

            obj_params = {}
            obj_params["N"] = num_spins
            obj_params["dissipation"] = dissipation
            obj_params["G"] = op

            grad_of_rho = np.zeros(num_params)
            params = center
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            mdic = {"rho": rho}
            outputfile = "QFI_test_center.mat"
            savemat(outputfile, mdic)

            for direction in range(1, num_params + 1):
                ## ONE FORWARD
                grad_of_rho = np.zeros(num_params)
                params = center
                params = params + h * B[direction - 1]
                rho = obj(params, num_spins, dissipation_rates=dissipation)

                mdic = {"rho": rho}
                outputfile = "QFI_test_pos_" + str(direction) + ".mat"
                savemat(outputfile, mdic)

                ## TWO FORWARD

                #grad_of_rho = np.zeros(num_params)
                #params = center
                #params = params + 2.0 * h * B[direction - 1]
                #rho = obj(params, num_spins, dissipation_rates=dissipation)

                #mdic = {"rho": rho}
                #outputfile = "QFI_test_2pos_" + str(direction) + ".mat"
                #savemat(outputfile, mdic)

                ## ONE BACKWARD
                grad_of_rho = np.zeros(num_params)
                params = center
                params = params - h * B[direction - 1]
                rho = obj(params, num_spins, dissipation_rates=dissipation)

                mdic = {"rho": rho}
                outputfile = "QFI_test_neg_" + str(direction) + ".mat"
                savemat(outputfile, mdic)

                ## TWO BACKWARD
                #grad_of_rho = np.zeros(num_params)
                #params = center
                #params = params - 2.0 * h * B[direction - 1]
                #rho = obj(params, num_spins, dissipation_rates=dissipation)

                #mdic = {"rho": rho}
                #outputfile = "QFI_test_2neg_" + str(direction) + ".mat"
                #savemat(outputfile, mdic)

                #spin_models.print_jacobian(grad_of_rho, precision=print_precision)
                #vals, vecs = compute_eigendecomposition(rho)

                #qfi, qfi_grad = compute_QFI_diffrax(vals, vecs, rho, params, grad_of_rho, get_jacobian, obj_params)
                #print(f"QFI is {qfi} for {params}")

