#!/usr/bin/env python3
import numpy as np
import PermSolver_methods as psm
import PermSolver_matrix as matrix
import random
import sys
from LBFGSB import LBFGSB

def LBFGSB_wrapper(x, obj_params, get_jacobian):

    x = np.squeeze(x.T)
    dim = len(x)
    qfi_grad = np.zeros(dim)

    rho = psm.simulate_layers(params=x,
                                     num_qubits=obj_params['N'],
                                     Hamiltonian_set=obj_params['Hmat_set'],
                                     dissipation_rates=obj_params['dissipation'] / np.pi)

    jacobian = get_jacobian(params=x,
                            num_qubits=obj_params['N'],
                            Hamiltonian_set=obj_params['Hmat_set'],
                            dissipation_rates=obj_params['dissipation'] / np.pi)

    qfi, new_grad = psm.compute_QFI(rho, x, jacobian, obj_params=obj_params, grad=qfi_grad)

    new_grad = np.expand_dims(new_grad, 0).T

    return -1.0 * qfi, -1.0 * new_grad

N = int(sys.argv[1])
model = sys.argv[2]
exec(f"Hmat_set = matrix.{model}Mat(1.0, N//2)")
dissipation = float(sys.argv[3])
layers = int(sys.argv[4])

obj_params = {'G': psm.MatSz(N//2), 'N': N, 'dissipation': dissipation, 'layers': layers, 'Hmat_set': Hmat_set}

random.seed(888)
np.random.seed(888)
# set up initial vector, parameter bounds
#x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * obj_params['layers'])] + [1])
#      * np.random.rand(3 + 2 * obj_params['layers']))
#bounds = [(0, 1/2) for _ in range(2)] + [(0,1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * obj_params['layers'])] + [(0, 1)]

num_params = 3 + 2 * layers
x0 = np.random.rand(num_params)

get_jacobian = psm.get_jacobian_func(psm.simulate_layers)  # check if simulate_layers is the correct string here.

func = lambda x: LBFGSB_wrapper(x, obj_params, get_jacobian)

lower_bounds = np.expand_dims(-np.inf * np.ones(num_params), 0).T
upper_bounds = np.expand_dims(np.inf * np.ones(num_params), 0).T
x0 = np.expand_dims(x0, 0).T
x, xhist, exitflag = LBFGSB(func, x0, lower_bounds, upper_bounds, m=10, tol=1e-5, max_iters=50, display=True, xhistory=False)
print(exitflag)
