#!/usr/bin/env python3
import matplotlib.pyplot as plt
import nlopt
import jax
import jax.numpy as jnp

import run_OAT_jax as run_OAT
from calculate_qfi_example import compute_QFI


def nlopt_wrapper(x, grad, obj_params):
    # x = jnp.array([0.5, x[0], x[1], 0])
    rho = run_OAT.simulate_OAT( x, obj_params["N"], obj_params["noise"])
    qfi = compute_QFI(rho, obj_params["G"])
    print("x:",x, " qfi:",qfi)
    return -1 * qfi  # negative because we are maximizing


if __name__ == "__main__":
    N = 4
    noise = 0.0
    G = run_OAT.collective_op(run_OAT.pauli_Z, N) / (2 * N)

    obj_params = {}  # Additional objective parameters
    obj_params["N"] = N
    obj_params["noise"] = noise
    obj_params["G"] = G

    n = 4
    opt = nlopt.opt(nlopt.LN_NELDERMEAD, n)  # Doesn't use derivatives and will work
    # opt = nlopt.opt(nlopt.LD_MMA, n) # Needs derivatives to work. Without grad being set (in-place) it is zero, so first iterate is deemed stationary

    opt.set_min_objective(lambda x, grad: nlopt_wrapper(x, grad, obj_params))

    lb = jnp.zeros(n)
    ub = jnp.ones(n)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    opt.set_xtol_rel(1e-4)

    #jnp.random.seed(1)
    #x0 = jnp.random.uniform(lb, ub, n)
    x0 = jax.random.uniform(jax.random.PRNGKey(1), shape=(n,), minval=lb, 
                             maxval=ub, dtype=jnp.float64)
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    print("optimum at ", x)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())
