#!/usr/bin/env python3
import os
import pickle
import sys

import nlopt
import numpy as np
from ibcdfo.pounders import pounders
from ibcdfo.pounders.general_h_funs import identity_combine as combinemodels
from mpi4py import MPI

import qfi_opt
from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_eigendecompotion, compute_QFI

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

orbit_dir = os.path.isdir(os.path.join(root_dir, "orbit", "py"))
orbit_found = os.path.isdir(orbit_dir)
if not orbit_found:
    messages = [
        "WARNING: 'orbit' not found",
        "If you already have orbit somewhere, run:",
        f"\n  ln -s <orbit-path> {root_dir}/orbit\n",
    ]
    print("\n".join(messages))
else:
    sys.path.append(orbit_dir)
    from orbit4py import ORBIT2


def identity_fun(arg):
    return arg


def sim_wrapper(x, grad, obj, obj_params):
    """Wrapper for `nlopt` that creates and updates a database of simulation inputs/outputs.

    Note that for large databases (or fast simulations), the database lookup can be more expensive than performing the simulation.
    """
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

    vals, vecs = compute_eigendecompotion(rho)
    qfi = compute_QFI(vals, vecs, obj_params["G"])
    print(x, qfi, flush=True)
    all_f.append(qfi)
    all_X.append(x)
    return -1 * qfi  # negative because we are maximizing


def calfun(x):
    return sim_wrapper(x, [], obj, obj_params)


def run_orbit(obj, obj_params, n, x0):
    gtol = 1e-9  # Gradient tolerance used to stop the local minimization [1e-5]
    rbftype = "cubic"  # Type of RBF (multiquadric, cubic, Gaussian) ['cubic']
    npmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
    trnorm = 0  # Type f trust-region norm [0]
    Low = -5000 * np.ones(n)  # 1-by-n Vector of lower bounds [zeros(1,n)]
    Upp = 5000 * np.ones(n)  # 1-by-n Vector of upper bounds [ones(1,n)]
    gamma_m = 0.5  # Reduction factor = factor of the LHS points you'd start a local run from [.5]
    maxdelta = np.inf
    delta = 1
    nfs = 1

    xkin = 0
    X = np.array(x0)
    F = np.array(calfun(X))

    [X, F, xkin, nf, exitflag, xkin_mat, xkin_val] = ORBIT2(
        calfun, rbftype, gamma_m, n, max_evals, npmax, delta, maxdelta, trnorm, gtol, Low, Upp, nfs, X, F, xkin
    )

    # print("optimum at ", X[xkin])
    # print("minimum value = ", F[xkin])

    return F[xkin], X[xkin]


def run_pounder(obj, obj_params, n, x0):
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

    [X, F, flag, xkin] = pounders(calfun, X, n, mpmax, max_evals, gtol, delta, nfs, m, F, xind, Low, Upp, printf, spsolver, identity_fun, combinemodels)

    # print("optimum at ", X[xkin])
    # print("minimum value = ", F[xkin])

    return F[xkin], X[xkin]


def run_nlopt(obj, obj_params, num_params, x0, solver):
    opt = nlopt.opt(getattr(nlopt, solver), num_params)
    # opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_params)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, num_params) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    opt.set_min_objective(lambda x, grad: sim_wrapper(x, grad, obj, obj_params))
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(max_evals)

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
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 4
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    for dissipation_rate in np.append([0], np.linspace(0.1, 5, 20)):
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 10

        for num_params in [4, 5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            for seed in range(size):
                if seed % size == rank:
                    # x0 = 0.5 * np.ones(num_params)  # This is an optimum for the num_params==4 problems
                    np.random.seed(seed)
                    x0 = np.random.uniform(lb, ub, num_params)

                    match num_params:
                        case 4:
                            models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
                            # models = ["simulate_OAT"]
                        case 5:
                            models = ["simulate_TAT", "simulate_local_TAT_chain"]
                            # models = ["simulate_TAT"]

                    for model in models:
                        filename = model + "_" + str(dissipation_rate) + "_" + str(seed) + ".pkl"
                        fig_filename = "Results_" + model + "_" + str(dissipation_rate) + "_" + str(seed)
                        if os.path.exists(filename):
                            continue
                        if os.path.exists(fig_filename + ".png"):
                            continue
                        obj = getattr(spin_models, model)

                        best_val = {}
                        best_pt = {}
                        for solver in ["LN_NELDERMEAD", "LN_BOBYQA", "ORBIT", "POUNDER"]:
                            global all_f, all_X
                            all_f = []
                            all_X = []
                            if solver in ["LN_NELDERMEAD", "LN_BOBYQA"]:
                                run_nlopt(obj, obj_params, num_params, x0, solver)
                            elif solver in ["ORBIT"]:
                                if not orbit_found:
                                    continue
                                run_orbit(obj, obj_params, num_params, x0)
                            elif solver in ["POUNDER"]:
                                run_pounder(obj, obj_params, num_params, x0)

                            print(all_f, all_X, flush=True)

                            ind = np.argmax(all_f)
                            best_val[solver] = all_f[ind]
                            best_pt[solver] = all_X[ind]

                        best_method = max(best_val, key=best_val.get)
                        # print(model, d, seed, best_val[best_method], best_pt[best_method], flush=True)
                        dic = {}
                        dic["best_method"] = best_method
                        dic["best_val"] = best_val[best_method]
                        dic["best_pt"] = best_pt[best_method]

                        with open(filename, "wb") as f:
                            pickle.dump(dic, f)

                        sys.exit("a")

                        #     plt.figure(fig_filename)
                        #     plt.plot(all_f, label=solver)

                        #     for i in range(1, len(all_f)):
                        #         all_f[i] = max(all_f[i - 1], all_f[i])

                        #     plt.figure(fig_filename + "best")
                        #     plt.plot(all_f, label=solver)

                        # plt.figure(fig_filename)
                        # plt.legend()
                        # plt.title(fig_filename)
                        # plt.savefig(fig_filename + ".png", dpi=300)

                        # plt.figure(fig_filename + "best")
                        # plt.legend()
                        # plt.title(fig_filename)
                        # plt.savefig(fig_filename + "best" + ".png", dpi=300)
