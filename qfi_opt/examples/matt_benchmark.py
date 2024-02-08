#!/usr/bin/env python3
import os
import pickle
import sys

import ipdb

import nlopt
import numpy as np
from ibcdfo.pounders import pounders
from ibcdfo.pounders.general_h_funs import identity_combine as combinemodels
#from mpi4py import MPI

sys.path.append('../../')

import qfi_opt
from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_eigendecomposition, compute_QFI

root_dir = os.path.dirname(os.path.dirname(qfi_opt.__file__))
minq5_dir = os.path.join(root_dir, "minq", "py", "minq5")
if not os.path.isdir(minq5_dir):
    messages = [
        "Please install (or symlink) MINQ in the QFI-Opt project directory.",
        "You can do this with:",
        f"\n  git clone https://github.com/POptUS/MINQ.git {root_dir}/minq\n",
        "Or, if you already have MINQ somewhere, run:",
        f"\n  ln -s <minq-path> {root_dir}/minq\n",
    ]
    exit("\n".join(messages))
sys.path.append(minq5_dir)


def sim_wrapper(x, h, qfi_grad, obj, obj_params):

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

    num_parameters = len(x)
    dA = np.zeros((num_parameters, N ** 2, N ** 2), dtype="cdouble")
    d2A = np.zeros((num_parameters, N ** 2, N ** 2), dtype="cdouble")
    if qfi_grad.size > 0:
        # Approximate the gradient
        I = np.eye(num_parameters)

        for parameter in range(num_parameters):
            # compute dA with respect to parameter
            rho_p = obj(x + h * I[parameter], obj_params["N"], dissipation_rates=obj_params["dissipation"])
            rho_m = obj(x - h * I[parameter], obj_params["N"], dissipation_rates=obj_params["dissipation"])
            dA[parameter] = (rho_p - rho_m) / (2 * h)
            # compute d2A with respect to parameter
            d2A[parameter] = (rho_p - 2 * rho + rho_m) / (h ** 2)

    # Compute eigendecomposition
    vals, vecs = compute_eigendecomposition(rho)

    qfi, new_grad = compute_QFI(vals, vecs, obj_params["G"], rho, dA, d2A, qfi_grad)
    print(x, qfi, new_grad, flush=True)

    try:
        if qfi_grad.size > 0:
            qfi_grad[:] = -1.0 * new_grad
    except:
        qfi_grad[:] = []

    return -1.0 * qfi #, -qfi_grad  # negative because we are maximizing
    #else:
    #    return -1 * qfi

def quick_test_gradient(obj, obj_params, x):
    h = 1e-8
    calfun = lambda x: sim_wrapper(x, h, True, obj, obj_params)
    calfun_f = lambda x: sim_wrapper(x, h, False, obj, obj_params)
    f0, g = calfun(x)
    print("f0: ", f0)

    # check for descent in gradient direction
    for j in range(4, 18, 2):
        stepsize = 10 ** (-j)
        xp = x - stepsize * g
        fp = calfun_f(xp)
        print("fp: ", fp, "stepsize: ", stepsize)




def run_pounder(obj, obj_params, n, x0):
    h = 1e-6
    calfun = lambda x: sim_wrapper(x, h, False, obj, obj_params)
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

def run_nlopt(obj, obj_params, num_params, x0, solver):
    opt = nlopt.opt(getattr(nlopt, solver), num_params)
    # opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_params)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, num_params) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    h = 1e-5
    opt.set_min_objective(lambda x, grad: sim_wrapper(x, h, grad, obj, obj_params))
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(300)
    opt.set_lower_bounds(-10.0 * np.ones(num_params))
    opt.set_upper_bounds(10.0 * np.ones(num_params))
    opt.set_vector_storage(0)


    # # Because the objective is periodic, don't set bounds (but don't need to sample so much)
    # opt.set_lower_bounds(lb)
    # opt.set_upper_bounds(ub)

    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    # print("optimum at ", x)
    # print("minimum value = ", minf)
    # print("result code = ", opt.last_optimize_result())

    return minf, x


if __name__ == "__main__":  # noqa: C901 # ignore "complexity" check for the code below
    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    size = 1

    N = 4
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    for dissipation_rate in np.append([0], np.linspace(0.1, 5, 20)):
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 300

        seed = 88
        np.random.seed(seed)

        for num_params in [4, 5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            #x0 = 0.5 * np.ones(num_params)
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
                minf, xfinal = run_nlopt(obj, obj_params, num_params, x0, "LD_LBFGS")
                # h = 1e-6
                # calfun = lambda x, grad: sim_wrapper(x, h, grad, obj, obj_params)
                #minf, xfinal = run_nlopt(obj, obj_params, num_params, x0, "LN_BOBYQA")
                #grad = np.zeros(num_params)
                #calfun(xfinal, grad)
