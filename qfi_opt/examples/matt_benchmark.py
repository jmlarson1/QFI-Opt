#!/usr/bin/env python3
import os
import sys

import nlopt
import numpy as np
from ibcdfo.pounders import pounders
from ibcdfo.pounders.general_h_funs import identity_combine as combinemodels
from scipy.io import savemat
#from mpi4py import MPI

import qfi_opt
from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_eigendecomposition, compute_QFI_diffrax, compute_QFI


def sim_wrapper(x, obj, obj_params):
    use_DB = False
    match = 0
    if use_DB:
        # Look through the database to see if there is a match
        database = obj.__name__ + "_" + str(obj_params["N"]) + "_" + str(obj_params["dissipation"]) + "_database.npy"
        DB = []
        if os.path.exists(database):
            DB = np.load(database, allow_pickle=True)
            for db_entry in DB:
                if np.allclose(db_entry["var_vals"], x, rtol=1e-12, atol=1e-12):
                    rho = db_entry["rho"]
                    match = 1
                    break

    if match == 0:
        # Do the sim
        rho = obj(x, obj_params["N"], dissipation_rates=obj_params["dissipation"])

        if use_DB:
            # Update database
            to_save = {"rho": rho, "var_vals": x}
            DB = np.append(DB, to_save)
            np.save(database, DB)

    # Compute eigendecomposition
    vals, vecs = compute_eigendecomposition(rho)

    qfi = compute_QFI(vals, vecs, obj_params["G"])
    print(x, qfi, flush=True)

    return -1.0 * qfi


def sim_wrapper_diffrax(x, qfi_grad, obj, obj_params, get_jacobian):

    use_DB = False
    match = 0
    if use_DB:
        # Look through the database to see if there is a match
        database = obj.__name__ + "_" + str(obj_params["N"]) + "_" + str(obj_params["dissipation"]) + "_database.npy"
        DB = []
        if os.path.exists(database):
            DB = np.load(database, allow_pickle=True)
            for db_entry in DB:
                if np.allclose(db_entry["var_vals"], x, rtol=1e-12, atol=1e-12):
                    rho = db_entry["rho"]
                    match = 1
                    break

    if match == 0:
        # Do the sim
        rho = obj(x, obj_params["N"], dissipation_rates=obj_params["dissipation"])

        if use_DB:
            # Update database
            to_save = {"rho": rho, "var_vals": x}
            DB = np.append(DB, to_save)
            np.save(database, DB)

    # force Hermitianness:
    rho = (rho + rho.conj().T) / 2.0
    # Compute eigendecomposition
    vals, vecs = compute_eigendecomposition(rho)

    qfi, new_grad = compute_QFI_diffrax(vals, vecs, x, qfi_grad, get_jacobian, obj_params)
    print(x, qfi, new_grad, flush=True)

    try:
        if qfi_grad.size > 0:
            qfi_grad[:] = -1.0 * new_grad
    except:
        qfi_grad[:] = []

    return -1.0 * qfi



def run_pounder(obj, obj_params, n, x0):
    calfun = lambda x: sim_wrapper(x, obj, obj_params)
    X = np.array(x0)
    F = np.array(calfun(X))
    Low = -np.inf * np.ones((1, n))
    Upp = np.inf * np.ones((1, n))
    mpmax = 2 * n + 1
    delta = 0.1
    m = 1
    nfs = 1
    printf = False
    spsolver = 2
    gtol = 1e-9
    xind = 0
    hfun = lambda F: F

    X, F, _, xkin = pounders(calfun, X, n, mpmax, max_evals, gtol, delta, nfs, m, F, xind, Low, Upp, printf, spsolver, hfun, combinemodels)

    # print("optimum at ", X[xkin])
    # print("minimum value = ", F[xkin])

    return F[xkin], X[xkin]

def run_nlopt(obj, obj_params, num_params, x0, solver, get_jacobian=False):
    opt = nlopt.opt(getattr(nlopt, solver), num_params)

    if not get_jacobian:
        opt.set_min_objective(lambda x, grad: sim_wrapper(x, obj, obj_params))
    else:
        opt.set_min_objective(lambda x, grad: sim_wrapper_diffrax(x, grad, obj, obj_params, get_jacobian))
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(300)
    opt.set_lower_bounds(-10.0 * np.ones(num_params))
    opt.set_upper_bounds(10.0 * np.ones(num_params))
    #opt.set_vector_storage(1)

    # # Because the objective is periodic, don't set bounds (but don't need to sample so much)
    # opt.set_lower_bounds(lb)
    # opt.set_upper_bounds(ub)

    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    # print("optimum at ", x)
    # print("minimum value = ", minf)

    return minf, x


if __name__ == "__main__":

    N = 4
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    for dissipation_rate in [1.0]:#np.linspace(0.1, 5, 20):
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 50

        seed = 88
        np.random.seed(seed)

        for num_params in [5]:#[4, 5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            x0 = np.random.uniform(lb, ub, num_params)

            match num_params:
                case 4:
                    models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
                    # models = ["simulate_OAT"]
                case 5:
                    models = ["simulate_TAT", "simulate_local_TAT_chain"]
                    # models = ["simulate_TAT"]

            for model in models:
                obj = getattr(spin_models, model)

                get_jacobian = spin_models.get_jacobian_func(obj)
                minf, xfinal = run_nlopt(obj, obj_params, num_params, x0, "LD_LBFGS", get_jacobian)
                #minf, xfinal = run_nlopt(obj, obj_params, num_params, x0, "LN_BOBYQA")
