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
    minimize_norm_diff,
    vec_compute_QFI_max_sum_squares,
    vec_compute_QFI_max_sum_squares_all,
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
    global all_f, all_X, all_eigvals, all_eigvecs, all_rho
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
        all_X.append(x)
        return -1 * qfi  # negative because we are maximizing
    elif out_type == 1:
        vecqfi = vec_compute_QFI_max_sum_squares(vals, vecs, obj_params["G"])
        all_f.append(np.sum(vecqfi**2))
        return vecqfi
    elif out_type == 2:
        vecqfi = vec_compute_QFI_more_struct_1(vals, vecs, obj_params["G"])
        all_f.append(h_more_struct_1(vecqfi))
        return vecqfi
    elif out_type == 3:
        if len(all_X):
            closest = np.inf
            for i, x1 in enumerate(all_X):
                norm_val = np.linalg.norm(x1 - x)
                if norm_val < closest:
                    best_ind = i

            perm = minimize_norm_diff(vecs, all_eigvecs[best_ind])
            assert np.linalg.norm(vecs - all_eigvecs[best_ind], "fro") >= np.linalg.norm(vecs @ perm - all_eigvecs[best_ind], "fro"), "Things got worse!"
            vecs = vecs @ perm
            vals = vals @ perm

        vecqfi = vec_compute_QFI_more_struct_1(vals, vecs, obj_params["G"])

        all_X.append(x)
        all_rho.append(rho)
        all_eigvals.append(vals)
        all_eigvecs.append(vecs)
        all_f.append(h_more_struct_1(vecqfi))
        return vecqfi
    elif out_type == 4:
        if len(all_X):
            closest = np.inf
            for i, x1 in enumerate(all_X):
                norm_val = np.linalg.norm(x1 - x)
                if norm_val < closest:
                    best_ind = i

            perm = minimize_norm_diff(vecs, all_eigvecs[best_ind])
            assert np.linalg.norm(vecs - all_eigvecs[best_ind], "fro") >= np.linalg.norm(vecs @ perm - all_eigvecs[best_ind], "fro"), "Things got worse!"
            vecs = vecs @ perm
            vals = vals @ perm

        vecqfi = vec_compute_QFI_max_sum_squares(vals, vecs, obj_params["G"])
        all_f.append(np.sum(vecqfi**2))
        all_X.append(x)
        all_rho.append(rho)
        all_eigvals.append(vals)
        all_eigvecs.append(vecs)
        return vecqfi
    elif out_type == 5:
        vecqfi = vec_compute_QFI_max_sum_squares_all(vals, vecs, obj_params["G"])
        all_f.append(np.sum(vecqfi**2))
        return vecqfi


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
    Ffun = lambda x: sim_wrapper(x, [], obj, obj_params, out_type=out_type)
    if out_type == 0:
        hfun = lambda F: F
        combinemodels = identity_combine
        m = 1
    elif out_type == 1 or out_type == 4 or out_type == 5:
        hfun = lambda F: -1 * np.sum(F**2)
        combinemodels = neg_leastsquares
        m = 120
        # m = 6
    elif out_type == 2 or out_type == 3:
        hfun = lambda F: -1 * h_more_struct_1(F)
        combinemodels = h_more_struct_1_combine
        m = 16 + 16 * 15  # 16 for the eigenvalues, 16*15/2 for the eigenvector pair real part and 16*15/2 for the imag part
        # m = 4 + 4 * 3  # 4 for the eigenvalues, 4*3/2 for the eigenvector pair real part and 4*3/2 for the imag part

    # # Let's make sure all the ways of calculating QFI produce the same result:
    # truth = sim_wrapper(x0, [], obj, obj_params, out_type = 0)
    # test_1 = -1*np.sum(sim_wrapper(x0, [], obj, obj_params, out_type = 1)**2)
    # test_2 = h_more_struct_1(sim_wrapper(x0, [], obj, obj_params, out_type = 2))
    # test_3 = h_more_struct_1(sim_wrapper(x0, [], obj, obj_params, out_type = 3))
    # import ipdb; ipdb.set_trace(context=21)

    X = np.array(x0)
    F = np.array(Ffun(X))
    Low = -np.inf * np.ones((1, n))
    Upp = np.inf * np.ones((1, n))
    delta_0 = 0.001
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

    for dissipation_rate in [0.0]:
        obj_params = {}
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation_rate
        obj_params["G"] = G

        max_evals = 25

        for num_params in [5]:
            lb = np.zeros(num_params)
            ub = np.ones(num_params)

            for seed in [0, 1, 2]:
                # x0 = 0.5 * np.ones(num_params)  # This is an optimum for the num_params==4 problems
                np.random.seed(seed)
                x0 = np.random.uniform(lb, ub, num_params)

                match num_params:
                    case 2:
                        models = ["simulate_OAT"]
                    case 4:
                        models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
                    case 5:
                        models = ["simulate_TAT", "simulate_local_TAT_chain"]

                for model in models:
                    print(model)
                    fig_filename = "Results_" + model + "_" + str(dissipation_rate) + "_" + str(seed)
                    # if os.path.exists(fig_filename + ".png"):
                    #     continue
                    obj = getattr(spin_models, model)

                    # for solver in ["LN_NELDERMEAD", "LN_BOBYQA", "POUNDER", "POUNDERS"]:
                    for number, solver in enumerate(["POUNDER", "POUNDERS1", "POUNDERS2"]):
                    # for number, solver in enumerate(["POUNDERS2"]):
                        global all_f, all_X, all_eigvals, all_eigvecs, all_rho
                        all_f = []
                        all_X = []
                        all_eigvals = []
                        all_eigvecs = []
                        all_rho = []
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
                        elif solver in ["POUNDERS3"]:
                            run_pounder(obj, obj_params, num_params, x0, 3)
                        elif solver in ["POUNDERS4"]:
                            run_pounder(obj, obj_params, num_params, x0, 4)
                        elif solver in ["POUNDERS5"]:
                            run_pounder(obj, obj_params, num_params, x0, 5)
                        else:
                            raise ValueError(f"Unknown solver: {solver}")

                        # print(all_X[0], all_X[-1])

                        # center = all_X[0]
                        # endpoint = all_X[-1]
                        # v = np.linspace(0,1,50)
                        # X = np.array([center*(1-i) + i*endpoint for i in v])
                        # fvals = np.zeros(len(X))

                        # rho = obj(center, obj_params["N"], dissipation_rates=obj_params["dissipation"])
                        # vals, vecs = compute_eigendecompotion(rho)
                        # qfi = compute_QFI(vals, vecs, obj_params["G"])
                        # print(qfi,"start")

                        # rho = obj(endpoint, obj_params["N"], dissipation_rates=obj_params["dissipation"])
                        # vals, vecs = compute_eigendecompotion(rho)
                        # qfi = compute_QFI(vals, vecs, obj_params["G"])
                        # print(qfi,"end")

                        # for i,x in enumerate(X):
                        #     rho = obj(x, obj_params["N"], dissipation_rates=obj_params["dissipation"])
                        #     vals, vecs = compute_eigendecompotion(rho)
                        #     qfi = compute_QFI(vals, vecs, obj_params["G"])
                        #     fvals[i] = qfi
                        #     print(x,fvals)

                        # import matplotlib.pyplot as plt
                        # plt.plot(fvals)
                        # plt.savefig('fvals.png',dpi=300)
                        # plt.close()
                        # sys.exit("a")

                        plt.figure(fig_filename)
                        if number % 2 == 0:
                            plt.plot(all_f, label=solver, linestyle="solid")
                        else:
                            plt.plot(all_f, label=solver, linestyle="dashed")

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
                    sys.exit("a")
