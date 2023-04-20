import os

import matplotlib.pyplot as plt
import nlopt
import numpy as np

import spin_models
from calculate_qfi_example import compute_QFI


def nlopt_wrapper(x, grad, obj, obj_params):
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

    qfi = compute_QFI(rho, obj_params["G"])
    print(x, qfi, flush=True)
    all_f.append(qfi)
    return -1 * qfi  # negative because we are maximizing


def run_nlopt(obj, obj_params, num_params, solver):

    opt = nlopt.opt(getattr(nlopt, solver), num_params)
    # opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_params)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, num_params) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    opt.set_min_objective(lambda x, grad: nlopt_wrapper(x, grad, obj, obj_params))
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(500)

    lb = np.zeros(num_params)
    ub = np.ones(num_params)
    # x0 = 0.5 * np.ones(num_params)  # This is an optimum for the num_params==4 problems
    np.random.seed(1)
    x0 = np.random.uniform(lb, ub, num_params)

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

    obj_params = {}
    obj_params["N"] = N
    obj_params["dissipation"] = 0
    obj_params["G"] = G

    for num_params in [4, 5]:
        match num_params:
            case 4:
                models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
            case 5:
                models = ["simulate_TAT", "simulate_local_TAT_chain"]

        for model in models:
            print(model)
            obj = getattr(spin_models, model)

            for solver in ["LN_NELDERMEAD", "LN_BOBYQA"]:
                filename = solver + "_" + model + ".txt"
                if not os.path.exists(filename):
                    global all_f
                    all_f = []
                    run_nlopt(obj, obj_params, num_params, solver)
                    np.savetxt(filename, all_f)
                else: 
                    all_f = np.loadtxt(filename)

                plt.plot(all_f, label=filename)

    plt.legend()
    plt.savefig("Results.png",dpi=300)
    

