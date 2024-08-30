#!/usr/bin/env python3
import nlopt
import numpy as np
import qfi_opt.spin_models as sm
from qfi_opt.examples import calculate_qfi as calc_qfi
from scipy.optimize import minimize as mini
from qfi_opt.examples.calculate_qfi import compute_eigendecomposition, compute_QFI
import random
import time
import sys

attempts = 10
# model = sys.argv[1]
# N = int(sys.argv[2])
# coupling_exponent = int(sys.argv[3])
# dissipation = float(sys.argv[4])
# layers = int(sys.argv[5])

dissipation = 0.01
model = 'local_TAT'
N = 4
coupling_exponent = 0
layers = 5

obj = getattr(sm, f'simulate_{model}_chain')
obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N)/(2*N), 'N': N, 'dissipation': dissipation,
              'coupling_exponent': coupling_exponent}

def min_func(x:np.ndarray, obj_params:dict)->float:
    rho = obj(params = x, num_qubits=obj_params['N'], dissipation_rates=obj_params['dissipation'], coupling_exponent=obj_params['coupling_exponent'])
    vals, vecs = calc_qfi.compute_eigendecomposition(rho)
    qfi = calc_qfi.compute_QFI(vals, vecs, x, obj_params=obj_params)[0]
    return -qfi

def sim_wrapper_nlopt(x, qfi_grad, obj, obj_params, get_jacobian):
    print(x)
    rho = obj(params = x, num_qubits=obj_params['N'], dissipation_rates=obj_params['dissipation'], coupling_exponent=obj_params['coupling_exponent'])
    vals, vecs = calc_qfi.compute_eigendecomposition(rho)
    qfi, new_grad = calc_qfi.compute_QFI(vals, vecs, x, obj_params=obj_params, grad=qfi_grad, get_jacobian=get_jacobian)

    try:
        if qfi_grad.size > 0:
            qfi_grad[:] = -1.0 * new_grad
    except:
        qfi_grad[:] = []

    return -1.0 * qfi


# set up initial vector, parameter bounds
x0, bounds = np.ones(3 + 2 * layers) * 1/2, [(0, 1/2) for _ in range(3 + 2 * layers)]

minimum = 0
num_params = 3 + 2 * layers

get_jacobian = sm.get_jacobian_func(obj)
# attempt optimization attempts * layers times, keep best
for attempt in range(attempts * layers):

    random.seed((time.time() * 10**7) % 10**7)
    x = x0 * np.random.rand(num_params)
    # out = mini(min_func, x, args=(obj_params, ), tol=1e-2, bounds=bounds, method='Nelder-Mead')

    opt = nlopt.opt(getattr(nlopt, "LD_LBFGS"), num_params)
    opt.set_min_objective(lambda x, grad: sim_wrapper_nlopt(x, grad, obj, obj_params, get_jacobian))
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(1000)
    opt.set_lower_bounds(0 * np.ones(num_params))
    opt.set_upper_bounds(0.5 * np.ones(num_params))
    xout = opt.optimize(x0)

    if out.fun < minimum:
        minimum = out.fun
        opt_qfi = out.fun
        opt_params = out.x


print(f'Optimal qfi found = {opt_qfi}, \nOptimal Params found = {opt_params}')






