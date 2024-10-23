#!/usr/bin/env python3
import numpy as np
import qfi_opt.spin_models as sm
from qfi_opt.examples import calculate_qfi as calc_qfi
import random
import time
import sys
import ipdb
from LBFGSB import LBFGSB

def LBFGSB_wrapper(x, obj, obj_params, get_jacobian):

    x = np.squeeze(x.T)
    dim = len(x)
    qfi_grad = np.zeros(dim)

    rho = obj(params=x, num_qubits=obj_params['N'], dissipation_rates=obj_params['dissipation'], coupling_exponent=obj_params['coupling_exponent'])
    vals, vecs = calc_qfi.compute_eigendecomposition(rho)
    # when you want to debug nonsmoothness, uncomment this line for sure:
    # print("params: ", x.T, "eigenvalues of rho(params): ", vals)
    qfi, new_grad = calc_qfi.compute_QFI(rho, vals, vecs, x, obj_params=obj_params, grad=qfi_grad, get_jacobian=get_jacobian)

    new_grad = np.expand_dims(new_grad, 0).T

    return -1.0 * qfi, -1.0 * new_grad

# N is number of spins, the examples we have use N = 4 or N = 5
#
# model should be a string argument.
# if N = 4, then valid arguments from our examples include
# "simulate_OAT", "simulate_ising_chain", "simulate_XX_chain"
# if N = 5, then valid arguments from our examples include
# "simulate_local_TAT_chain", "simulate_TAT"
#
# coupling exponent should be a nonnegative integer. 0 is hard, seems to induce more nonsmoothness,
# this hypothesis is worth studying.
#
# dissipation is a nonnegative float.
#
# layers should be a positive integer specifying the depth of a parameterized circuit.
# in the examples we were given, this only made sense for simulate_local_TAT_chain, and in that case,
# the number of parameters scales like 3 + 2 * layers


N = int(sys.argv[1])
model = sys.argv[2]
coupling_exponent = int(sys.argv[3])
dissipation = float(sys.argv[4])
layers = int(sys.argv[5])

obj = getattr(sm, f'{model}')
obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N)/(2*N), 'N': N, 'dissipation': dissipation,
              'coupling_exponent': coupling_exponent}

# set up initial vector, parameter bounds
x0, bounds = np.random.rand(3 + 2 * layers), [(0.0, 1.0) for _ in range(3 + 2 * layers)]

num_params = 3 + 2 * layers

get_jacobian = sm.get_jacobian_func(obj)

random.seed((time.time() * 10**7) % 10**7)

func = lambda x: LBFGSB_wrapper(x, obj, obj_params, get_jacobian)
lower_bounds = np.expand_dims(np.zeros(num_params), 0).T
upper_bounds = np.expand_dims(np.ones(num_params), 0).T
x0 = np.expand_dims(x0, 0).T
x, xhist = LBFGSB(func, x0, lower_bounds, upper_bounds, m=10, tol=1e-5, max_iters=50, display=True, xhistory=False)






