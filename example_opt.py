#!/usr/bin/env python3
import matplotlib.pyplot as plt
import nlopt
import numpy as np

import run_OAT
from calculate_qfi_example import compute_QFI


def nlopt_wrapper(x, grad, obj_params):
    # x = np.array([0.5, x[0], x[1], 0])
    rho = run_OAT.simulate_OAT(obj_params["N"], x, obj_params["noise"])
    qfi = compute_QFI(rho, obj_params["G"])
    print(x, qfi)
    return -1 * qfi  # negative because we are maximizing


if __name__ == "__main__":
    N = 4
    noise = 0
    G = run_OAT.collective_op(run_OAT.pauli_Z, N) / (2 * N)

    obj_params = {}  # Additional objective parameters
    obj_params["N"] = N
    obj_params["noise"] = noise
    obj_params["G"] = G

    n = 4
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, n)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, n) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    opt.set_min_objective(lambda x, grad: nlopt_wrapper(x, grad, obj_params))

    lb = np.zeros(n)
    ub = np.ones(n)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    opt.set_xtol_rel(1e-4)

    np.random.seed(1)
    x0 = np.random.uniform(lb, ub, n)
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    print("optimum at ", x)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())
