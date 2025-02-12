import numpy as np
import time, os, sys
np.random.seed(int((time.time() * 10 ** 8) % 10** 8))
from scipy.optimize import minimize as mini
import PermSolver_methods as methods
import PermSolver_matrix as matrix
import random

def min_funct(x0:np.ndarray, obj_params:dict, Hmat_set:list):

    rho = methods.simulate_layers(params=x0,
                                     num_qubits=obj_params['N'],
                                     Hamiltonian_set=Hmat_set,
                                     dissipation_rates=obj_params['dissipation'] / np.pi)

    qfi = methods.calc_QFI(rho, obj_params['N']//2, obj_params['G']) / obj_params['N']**2
    print(qfi)
    return -qfi


# set optimization parameters
method = 'Nelder-Mead' #"L-BFGS-B" # 'Nelder-Mead'
obj_params = {}
obj_params['N'] = 8
model = 'TAT' #np.random.choice(['OAT', 'TAT'])
obj_params['dissipation'] = 0.01  #float(np.random.choice(np.logspace(-2, 1.5, 7)))
obj_params['layers'] = 2
obj_params['G'] = methods.MatSz(obj_params['N']//2)
exec(f"Hmat_set = matrix.{model}Mat(1.0, obj_params['N']//2)")

x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * obj_params['layers'])] + [1])
      * np.random.rand(3 + 2 * obj_params['layers']))
bnds = [(0, 1/2) for _ in range(2)] + [(0,1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * obj_params['layers'])] + [(0, 1)]
# unbounded case:
bnds = [(-np.inf, np.inf) for _ in range(3 + 2 * obj_params['layers'])]

random.seed(888)
np.random.seed(888)
x0 = np.random.rand(3 + 2 * obj_params['layers'])

# optimize qfi
time_stamp = time.time()
out = mini(min_funct,
                  x0=x0,
                  method=method,
                  bounds=bnds,
                  tol=1e-2,
                  args=(obj_params, Hmat_set, ))

print(f'{model} runtime took {(time.time() - time_stamp)/60:.2f} minutes',
      f'qfi = {-out.fun:.3f}',
      sep='\n')



