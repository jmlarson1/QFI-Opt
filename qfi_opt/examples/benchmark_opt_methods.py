import os
import sys

import matplotlib.pyplot as plt
import nlopt
import numpy as np

from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import compute_QFI, compute_eigendecompotion

try:
    from ibcdfo.pounders import pounders
    from ibcdfo.pounders.general_h_funs import identity_combine as combinemodels
except:
    sys.exit("Please 'pip install ibcdfo'")

try:
    sys.path.append("../../../minq/py/minq5/")  # Needed by pounders, but not pip installable
    from minqsw import minqsw
except:
    sys.exit("Make sure the MINQ [https://github.com/POptUS/minq] is installed (or symlinked) in the same directory as your QFI-Opt package")

# sys.path.append("../orbit/py")
# from orbit4py import ORBIT2


def sim_wrapper(x, grad, obj, obj_params):
    """Wrapper for `nlopt` that creates and updates a database of simulation inputs/outputs.

    Note that for large databases (or fast simulations), the database lookup can be more expensive than performing the simulation.
    """
    global all_f
    database = obj.__name__ + "_" + str(obj_params["N"]) + "_" + str(obj_params["dissipation"]) + "_database.npy"
    DB = []
    match = 0
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

        to_save = {"rho": rho, "var_vals": x}
        DB = np.append(DB, to_save)
        np.save(database, DB)

    vals, vecs = compute_eigendecompotion(rho)
    qfi = compute_QFI(vals, vecs, obj_params["G"])
    print(x, qfi, flush=True)
    all_f.append(qfi)
    return -1 * qfi  # negative because we are maximizing


def run_orbit(obj, obj_params, n, x0):
    calfun = lambda x: sim_wrapper(x, [], obj, obj_params)
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


def run_pounder(obj, obj_params, n, x0):
    calfun = lambda x: sim_wrapper(x, [], obj, obj_params)
    X = np.array(x0)
    F = np.array(calfun(X))
    Low = -np.inf * np.ones((1, n))
    Upp = np.inf * np.ones((1, n))
    mpmax = 2 * n + 1
    delta = 0.1
    m = 1
    nfs = 1
    printf = True
    spsolver = 2
    gtol = 1e-9
    xind = 0
    hfun = lambda F: F

    [X, F, flag, xkin] = pounders(calfun, X, n, mpmax, max_evals, gtol, delta, nfs, m, F, xind, Low, Upp, printf, spsolver, hfun, combinemodels)


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
    print("optimum at ", x)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())


if __name__ == "__main__":
    N = 4
    G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

    for dissipation_rate in np.append([0], np.linspace(0.1, 5, 20)):
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 100

        for num_params in [4, 5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            for seed in [0, 1]:
                # x0 = 0.5 * np.ones(num_params)  # This is an optimum for the num_params==4 problems
                np.random.seed(seed)
                x0 = np.random.uniform(lb, ub, num_params)

                match num_params:
                    case 4:
                        models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
                    case 5:
                        models = ["simulate_TAT", "simulate_local_TAT_chain"]

                for model in models:
                    print(model)
                    fig_filename = "Results_" + model + "_" + str(dissipation_rate) + "_" + str(seed)
                    if os.path.exists(fig_filename + ".png"):
                        continue
                    obj = getattr(spin_models, model)

                    for solver in ["LN_NELDERMEAD", "LN_BOBYQA", "POUNDER"]:
                        global all_f
                        all_f = []
                        if solver in ["LN_NELDERMEAD", "LN_BOBYQA"]:
                            run_nlopt(obj, obj_params, num_params, x0, solver)
                        elif solver in ["ORBIT"]:
                            run_orbit(obj, obj_params, num_params, x0)
                        elif solver in ["POUNDER"]:
                            run_pounder(obj, obj_params, num_params, x0)

                        plt.figure(fig_filename)
                        plt.plot(all_f, label=solver)

                        for i in range(1, len(all_f)):
                            all_f[i] = max(all_f[i - 1], all_f[i])

                        plt.figure(fig_filename + "best")
                        plt.plot(all_f, label=solver)

                    plt.figure(fig_filename)
                    plt.legend()
                    plt.title(fig_filename)
                    plt.savefig(fig_filename + ".png", dpi=300)

                    plt.figure(fig_filename + "best")
                    plt.legend()
                    plt.title(fig_filename)
                    plt.savefig(fig_filename + "best" + ".png", dpi=300)
