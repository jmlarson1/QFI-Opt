import matplotlib.pyplot as plt
import nlopt

import numpy as np
import spin_models

from calculate_qfi_example import compute_QFI


def nlopt_wrapper(x, grad, obj, obj_params):
    rho = obj(x, obj_params["N"], dissipation_rates=obj_params["dissipation"])
    qfi = compute_QFI(rho, obj_params["G"])
    print(x, qfi, flush=True)
    return -1 * qfi  # negative because we are maximizing

def run_nlopt(obj, obj_params):
    N = obj_params['N']

    opt = nlopt.opt(nlopt.LN_NELDERMEAD, N)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, N) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    opt.set_min_objective(lambda x, grad: nlopt_wrapper(x, grad, obj, obj_params))
    opt.set_xtol_rel(1e-4)

    lb = np.zeros(N)
    ub = np.ones(N)
    x0 = 0.5 * np.ones(N) # This is the optimum for the N==4 problems 
    np.random.seed(1)
    x0 = np.random.uniform(lb, ub, N)

    # # Because the objective is periodic, don't set bounds (but don't need to sample so much)
    # opt.set_lower_bounds(lb)
    # opt.set_upper_bounds(ub)

    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    print("optimum at ", x)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())

if __name__ == "__main__":
    # Calculate QFI for models at random points in the domain.
    dissipation = 0

    for N in [4, 5]:
        G = spin_models.collective_op(spin_models.PAULI_Z, N) / (2 * N)

        obj_params = {}  # Additional objective parameters
        obj_params["N"] = N
        obj_params["dissipation"] = dissipation
        obj_params["G"] = G
        match N:
            case 4:
                models = ["simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"]
            case 5:
                models = ["simulate_TAT", "simulate_local_TAT_chain"]

        for model in models:
            print(model)
            obj = getattr(spin_models, model)

            run_nlopt(obj, obj_params)
