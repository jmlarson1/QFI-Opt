import os
import sys

import matplotlib.pyplot as plt
import nlopt
import numpy as np

from qfi_opt import spin_models
from qfi_opt.examples.calculate_qfi import (
    compute_eigendecompotion,
    compute_QFI,
    h_more_struct_1,
    h_more_struct_1_combine,
    vec_compute_QFI_max_sum_squares,
    vec_compute_QFI_more_struct_1,
)

try:
    from ibcdfo.pounders import pounders
    from ibcdfo.pounders.general_h_funs import identity_combine, neg_leastsquares
except:
    sys.exit("Please 'pip install ibcdfo'")

try:
    sys.path.append("../../../../poptus/minq/py/minq5/")  # Needed by pounders, but not pip installable
    from minqsw import minqsw
except:
    sys.exit("Make sure the MINQ [https://github.com/POptUS/minq] is installed (or symlinked) in the same directory as your QFI-Opt package")

# sys.path.append("../orbit/py")
# from orbit4py import ORBIT2


def sim_wrapper(x, grad, obj, obj_params, out_type=0):
    """Wrapper for `nlopt` that creates and updates a database of simulation inputs/outputs.

    Note that for large databases (or fast simulations), the database lookup can be more expensive than performing the simulation.
    """
    global all_f
    database = obj.__name__ + "_" + str(obj_params["N"]) + "_" + str(obj_params["dissipation"]) + "_database.npy"
    DB = []
    match = 0
    use_DB = True
    if use_DB and os.path.exists(database):
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
            to_save = {"rho": rho, "var_vals": x}
            DB = np.append(DB, to_save)
            np.save(database, DB)

    vals, vecs = compute_eigendecompotion(rho)
    if out_type == 0:
        qfi = compute_QFI(vals, vecs, obj_params["G"])
        print(x, qfi, flush=True)
        all_f.append(qfi)
        return -1 * qfi  # negative because we are maximizing
    elif out_type == 1:
        vecqfi = vec_compute_QFI_max_sum_squares(vals, vecs, obj_params["G"])
        all_f.append(np.sum(vecqfi**2))
        return vecqfi
    elif out_type == 2:
        vecqfi = vec_compute_QFI_more_struct_1(vals, vecs, obj_params["G"])
        all_f.append(qfi)
        return vecqfi  # negative because we are maximizing


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


def run_pounder(obj, obj_params, n, x0, out_type=0):
    if out_type == 0:
        Ffun = lambda x: sim_wrapper(x, [], obj, obj_params)
        hfun = lambda F: F
        combinemodels = identity_combine
        m = 1
    elif out_type == 1:
        Ffun = lambda x: sim_wrapper(x, [], obj, obj_params, out_type=1)
        hfun = lambda F: -1 * np.sum(F**2)
        combinemodels = neg_leastsquares
        m = 120
    elif out_type == 2:
        Ffun = lambda x: sim_wrapper(x, [], obj, obj_params, out_type=2)
        hfun = h_more_struct_1
        combinemodels = h_more_struct_1_combine
        m = 16 + 16 * 15  # 16 for the eigenvalues, 16*15/2 for the eigenvector pair real part and 16*15/2 for the imag part

    # Let's make sure all the ways of calculating QFI produce the same result:
    import ipdb; ipdb.set_trace(context=21)
    truth = sim_wrapper(x0, [], obj, obj_params, out_type = 0)
    test_1 = -1*np.sum(sim_wrapper(x0, [], obj, obj_params, out_type = 1)**2)
    test_2 = h_more_struct_1(sim_wrapper(x0, [], obj, obj_params, out_type = 2))

    X = np.array(x0)
    F = np.array(Ffun(X))
    Low = -np.inf * np.ones((1, n))
    Upp = np.inf * np.ones((1, n))
    delta_0 = 0.1
    g_tol = 1e-9

    Options = {"hfun": hfun, "combinemodels": combinemodels, "printf": True}
    Prior = {"X_init": X, "F_init": F, "nfs": 1, "xk_in": 0}
    [X, F, hF, flag, xkin] = pounders(Ffun, X, n, max_evals, g_tol, delta_0, m, Low, Upp, Prior=Prior, Options=Options)


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

    for dissipation_rate in [0.0, 0.1, 0.2]:
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 100

        for num_params in [4, 5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            for seed in [0, 1, 2]:
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

                    # for solver in ["LN_NELDERMEAD", "LN_BOBYQA", "POUNDER", "POUNDERS"]:
                    for solver in ["POUNDER", "POUNDERS1", "POUNDERS2"]:
                        global all_f
                        all_f = []
                        if solver in ["LN_NELDERMEAD", "LN_BOBYQA"]:
                            run_nlopt(obj, obj_params, num_params, x0, solver)
                        elif solver in ["ORBIT"]:
                            run_orbit(obj, obj_params, num_params, x0)
                        elif solver in ["POUNDER"]:
                            run_pounder(obj, obj_params, num_params, x0)
                        elif solver in ["POUNDERS1"]:
                            run_pounder(obj, obj_params, num_params, x0, 1)
                        elif solver in ["POUNDERS2"]:
                            run_pounder(obj, obj_params, num_params, x0, 2)

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
