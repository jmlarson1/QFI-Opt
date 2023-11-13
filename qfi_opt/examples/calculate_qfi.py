#!/usr/bin/env python3
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from qfi_opt import spin_models


def variance(rho: np.ndarray, G: np.ndarray) -> float:
    """Variance of self-adjoint operator (observable) G in the state rho."""
    return (G @ G @ rho).trace().real - (G @ rho).trace().real ** 2

def minimize_norm_diff(A, B):
    np.set_printoptions(precision=6, linewidth=400)

    n = A.shape[0]

    # Create a model
    m = gp.Model("matrix_permutation")

    # Create variables
    P = m.addVars(n, n, vtype=GRB.BINARY, name="P")


    # Add constraints for the permutation matrix
    for i in range(n):
        m.addConstr(gp.quicksum(P[i, j] for j in range(n)) == 1)
        m.addConstr(gp.quicksum(P[j, i] for j in range(n)) == 1)

    # AP_R = m.addVars(n, n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="AP_R")
    # AP_I = m.addVars(n, n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="AP_I")
    # for i in range(n):
    #     for j in range(n):
    #         m.addConstr(AP_R[i, j] == gp.quicksum(A[i, k].real * P[k, j] for k in range(n)))
    #         m.addConstr(AP_I[i, j] == gp.quicksum(A[i, k].imag * P[k, j] for k in range(n)))
    # obj = gp.quicksum((AP_R[i, j] - B[i, j].real)**2 + (AP_I[i, j] - B[i, j].imag)**2 for i in range(n) for j in range(n))

    obj = gp.quicksum((gp.quicksum(A[i, k].real * P[k, j] for k in range(n)) - B[i, j].real)**2 + (gp.quicksum(A[i, k].imag * P[k, j] for k in range(n)) - B[i, j].imag)**2 for i in range(n) for j in range(n))

    # Set the objective
    m.setObjective(obj, GRB.MINIMIZE)

    # Optimize the model
    m.optimize()

    # Extract the solution
    solution = np.zeros((n, n), dtype=int)
    if m.status == GRB.OPTIMAL:
        for i in range(n):
            for j in range(n):
                solution[i,j] = int(P[i,j].x > 0.5)

    # Apply the permutation to A
    # permuted_A = A[:, np.nonzero(solution)[1]]

    return solution

    # n, m = A.shape
    # if B.shape != (n, m):
    #     raise ValueError("Matrices A and B must have the same shape.")
    # # Create a new model
    # model = gp.Model("matrix_permutation")

    # # Create variables
    # # x[i, j] = 1 if column i of A is assigned to column j of B
    # x = model.addVars(m, m, vtype=GRB.BINARY, name="x")

    # # Each column in A is assigned to exactly one column in B
    # model.addConstrs((x.sum(i, '*') == 1 for i in range(m)), "row")
    # # Each column in B receives exactly one column from A
    # model.addConstrs((x.sum('*', j) == 1 for j in range(m)), "col")

    # # Objective: minimize the Frobenius norm of (A - B) 
    # objective = gp.quicksum(x[i, j] * np.linalg.norm(A[:, i] - B[:, j])**2 for i in range(m) for j in range(m))
    # model.setObjective(objective, GRB.MINIMIZE)

    # # Solve the model
    # model.optimize()

    # # Retrieve the solution
    # perm = np.zeros(m, dtype=int)
    # for i in range(m):
    #     for j in range(m):
    #         if x[i, j].X > 0.5:  # select the assigned columns
    #             perm[i] = j

    # assert np.linalg.norm(A-B,'fro') >= np.linalg.norm(A[:,perm] - B,'fro'), "Things got worse!"

    # return perm

def compute_eigendecompotion(rho: np.ndarray):
    # Compute eigendecomposition for rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    
    reg_term = min(eigvals)
    if reg_term < 0:
        eigvals, eigvecs = np.linalg.eigh(rho + -2*reg_term*np.eye(len(eigvals))) 

    assert min(eigvals) > 0

    return eigvals, eigvecs


def compute_QFI(eigvals: np.ndarray, eigvecs: np.ndarray, G: np.ndarray, tol: float = 1e-8, etol_scale: float = 10) -> float:
    num_vals = len(eigvals)

    # There should never be negative eigenvalues, so their magnitude gives an
    # empirical estimate of the numerical accuracy of the eigendecomposition.
    # We discard any QFI terms denominators within an order of magnitude of
    # this value.
    tol = max(tol, -etol_scale * np.min(eigvals))

    # Compute QFI
    running_sum = 0
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvecs[:,i].conj() @ G @ eigvecs[:,j]
                running_sum += numer / denom * np.linalg.norm(term) ** 2

    return 4 * running_sum


def vec_compute_QFI_max_sum_squares(eigvals: np.ndarray, eigvecs: np.ndarray, G: np.ndarray, tol: float = 1e-8, etol_scale: float = 10) -> float:
    # To be given to pounders, which will maximize sum squares of a vector input
    num_vals = len(eigvals)

    tol = max(tol, -etol_scale * np.min(eigvals))

    count = -1
    vecout = np.zeros(num_vals * (num_vals - 1) // 2)
    # Compute QFI
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            count += 1
            if not np.isclose(denom, 0, atol=tol, rtol=tol):
                numer = eigvals[i] - eigvals[j]
                term = eigvecs[:,i].conj() @ G @ eigvecs[:,j]
                vecout[count] = numer / np.sqrt(denom) * np.linalg.norm(term)

    return 2 * vecout


def h_more_struct_1(z):
    num_vals = 16
    b = num_vals * (num_vals - 1) // 2

    eigvals = z[:num_vals]
    eigvec_product_R = np.zeros((num_vals, num_vals))
    eigvec_product_I = np.zeros((num_vals, num_vals))

    count = -1
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            count += 1
            eigvec_product_R[i, j] = z[num_vals + count]
            eigvec_product_I[i, j] = z[num_vals + b + count]

    tol = max(1e-8, -10 * np.min(eigvals))

    running_sum = 0
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = eigvals[i] + eigvals[j]
            if not np.isclose(denom, 0, atol=tol, rtol=tol):
                numer = (eigvals[i] - eigvals[j]) ** 2
                term = eigvec_product_R[i, j] ** 2 + eigvec_product_I[i, j] ** 2
                running_sum += numer / denom * term
    return 4 * running_sum  # Negative because we are maximizing


def h_more_struct_1_combine(Cres, Gres, Hres):
    n, _, m = Hres.shape
    h1_g = np.zeros(n)
    h1_H = np.zeros((n, n))

    num_vals = 16
    b = num_vals * (num_vals - 1) // 2

    C_lam = Cres[:num_vals]
    G_lam = Gres[:, :num_vals]
    H_lam = Hres[:, :, :num_vals]

    C_vec_R = np.zeros((num_vals, num_vals))
    G_vec_R = np.zeros((n, num_vals, num_vals))
    H_vec_R = np.zeros((n, n, num_vals, num_vals))
    C_vec_I = np.zeros((num_vals, num_vals))
    G_vec_I = np.zeros((n, num_vals, num_vals))
    H_vec_I = np.zeros((n, n, num_vals, num_vals))

    count = -1
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            count += 1

            C_vec_R[i, j] = Cres[num_vals + count]
            G_vec_R[:, i, j] = Gres[:, num_vals + count]
            H_vec_R[:, :, i, j] = Hres[:, :, num_vals + count]

            C_vec_R[i, j] = Cres[num_vals + b + count]
            G_vec_R[:, i, j] = Gres[:, num_vals + b + count]
            H_vec_R[:, :, i, j] = Hres[:, :, num_vals + b + count]

    assert np.array_equal(C_vec_R.shape, C_vec_I.shape), "Should be the same size"
    assert np.array_equal(G_vec_R.shape, G_vec_I.shape), "Should be the same size"
    assert np.array_equal(H_vec_R.shape, H_vec_I.shape), "Should be the same size"

    tol = max(1e-8, -10 * np.min(C_lam))
    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            denom = C_lam[i] + C_lam[j]

            if not np.isclose(denom, 0, atol=tol, rtol=tol):
                T2 = C_vec_R[i, j] ** 2 + C_vec_I[i, j] ** 2
                T2_g = 2 * (C_vec_R[i, j] * G_vec_R[:, i, j] + C_vec_I[i, j] * G_vec_I[:, i, j])
                T2_H = 2 * (
                    C_vec_R[i, j] * H_vec_R[:, :, i, j]
                    + np.outer(G_vec_R[:, i, j], G_vec_R[:, i, j])
                    + np.outer(G_vec_I[:, i, j], G_vec_I[:, i, j])
                    + C_vec_I[i, j] * H_vec_I[:, :, i, j]
                )

                T1 = (C_lam[i] - C_lam[j]) ** 2 / (C_lam[i] + C_lam[j])

                N = 2 * (C_lam[i] ** 2 - C_lam[j] ** 2) * (G_lam[:, i] - G_lam[:, j]) - (C_lam[i] - C_lam[j]) ** 2 * (G_lam[:, i] + G_lam[:, j])

                N_g = (
                    2 * (C_lam[i] ** 2 - C_lam[j] ** 2) * (H_lam[:, :, i] - H_lam[:, :, j])
                    + 4 * np.outer(C_lam[i] * G_lam[:, i] - C_lam[j] * G_lam[:, j], G_lam[:, i] - G_lam[:, j])
                    - (C_lam[i] - C_lam[j]) ** 2 * (H_lam[:, :, i] + H_lam[:, :, j])
                    - 2 * (C_lam[i] - C_lam[j]) * np.outer(G_lam[:, i] - G_lam[:, j], G_lam[:, i] + G_lam[:, j])
                )

                T1_g = N / (C_lam[i] + C_lam[j]) ** 2
                T1_H = ((C_lam[i] + C_lam[j]) * N_g - 2 * np.outer(N, G_lam[:, i] + G_lam[:, j])) / (C_lam[i] + C_lam[j])

                h1_g += T1 * T2_g + T2 * T1_g
                h1_H += T1 * T2_H + np.outer(T1_g, T2_g) + np.outer(T2_g, T1_g) + T2 * T1_H

    return 4 * h1_g, 4 * h1_H


def vec_compute_QFI_more_struct_1(eigvals: np.ndarray, eigvecs: np.ndarray, G: np.ndarray, tol: float = 1e-8, etol_scale: float = 10) -> float:
    # To be given to pounders, which will maximize the mapping h_more_struct_1

    num_vals = len(eigvals)

    num_products = num_vals * (num_vals - 1) // 2
    eigvec_product_R = np.zeros(num_products)
    eigvec_product_I = np.zeros(num_products)
    count = -1

    for i in range(num_vals):
        for j in range(i + 1, num_vals):
            count += 1
            val = eigvecs[:,i].conj() @ G @ eigvecs[:,j]
            eigvec_product_R[count] = np.real(val)
            eigvec_product_I[count] = np.imag(val)

    assert count + 1 == num_products

    return np.hstack((eigvals, eigvec_product_R, eigvec_product_I))


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
            qfi2 = vec_compute_QFI_max_sum_squares(vals, vecs, op)
            qfi3 = vec_compute_QFI_more_struct_1(vals, vecs, op)
            print(f"QFI is {qfi} for {params}, or equivalently {np.sum(qfi2**2)}, or {h_more_struct_1(qfi3)}")

            params[-1] = 0.0
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            vals, vecs = compute_eigendecompotion(rho)
            qfi = compute_QFI(vals, vecs, op)
            qfi2 = vec_compute_QFI_max_sum_squares(vals, vecs, op)
            qfi3 = vec_compute_QFI_more_struct_1(vals, vecs, op)
            print(f"QFI is {qfi} for {params}, or equivalently {np.sum(qfi2**2)}, or {h_more_struct_1(qfi3)}")

            params[-1] = 1.0
            rho = obj(params, num_spins, dissipation_rates=dissipation)
            vals, vecs = compute_eigendecompotion(rho)
            qfi = compute_QFI(vals, vecs, op)
            qfi2 = vec_compute_QFI_max_sum_squares(vals, vecs, op)
            qfi3 = vec_compute_QFI_more_struct_1(vals, vecs, op)
            print(f"QFI is {qfi} for {params}, or equivalently {np.sum(qfi2**2)}, or {h_more_struct_1(qfi3)}")
