#!/usr/bin/env python3
import numpy as np
import qfi_opt.spin_models as sm
from qfi_opt.examples import calculate_qfi as calc_qfi
from scipy.optimize import minimize as mini
import random
import time
import sys

attempts = 10
model = sys.argv[1]
N = int(sys.argv[2])
coupling_exponent = int(sys.argv[3])
dissipation = float(sys.argv[4])
layers = int(sys.argv[5])

opt_qfis = np.zeros(len(layers))
obj = getattr(sm, f'simulate_{model}_chain')
obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N)/(2*N), 'N': N, 'dissipation': dissipation,
              'coupling_exponent': coupling_exponent}

def min_func(x:np.ndarray, obj_params:dict)->float:
    rho = obj(params = x, num_qubits=obj_params['N'], dissipation_rates=obj_params['dissipation'], coupling_exponent=obj_params['coupling_exponent'])
    vals, vecs = calc_qfi.compute_eigendecomposition(rho)
    qfi = calc_qfi.compute_QFI(vals, vecs, x, obj_params=obj_params)[0]
    return -qfi

# set up initial vector, parameter bounds
x0, bounds = np.ones(3 + 2 * layers) * 1/2, [(0, 1/2) for _ in range(3 + 2 * layers)]

minimum = 0
# attempt optimization attempts * layers times, keep best
for attempt in range(attempts * layers):

    random.seed((time.time() * 10**7) % 10**7)
    x = x0 * np.random.rand(3 + 2 * layers)
    out = mini(min_func, x, args=(obj_params, ), tol=1e-2, bounds=bounds, method='Nelder-Mead')
    if out.fun < minimum:
        minimum = out.fun
        opt_qfi = out.fun
        opt_params = out.x


print(f'Optimal qfi found = {opt_qfi}, \nOptimal Params found = {opt_params}')






