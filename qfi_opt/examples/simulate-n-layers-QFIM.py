import numpy as np
import qfi_opt.spin_models as sm
from qfi_opt.examples import calculate_qfi as calc_qfi
from scipy.optimize import minimize as mini
import random
import time
import sys

def compute_QFIM(vals:np.ndarray, vecs:np.ndarray, num_qubits:int)->[np.ndarray, np.ndarray]:
    def eigen(A):
        eigenValues, eigenVectors = np.linalg.eig(A)
        idx = np.argsort(eigenValues)
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        return eigenValues, eigenVectors

    ops = list(sm.collective_spin_ops(num_qubits))
    covar = np.zeros((3, 3))

    for idx_op1, op1 in enumerate(ops):
        for idx_op2, op2 in enumerate(ops):
            if idx_op2 >= idx_op1:
                covar[idx_op1, idx_op2] = 2 * sum([(vals[ii] - vals[jj])**2 * ((vecs[ii].T.conj() @ op1 @ vecs[jj]) *
                                          (vecs[jj].T.conj() @ op2 @ vecs[ii])).real/(vals[ii] + vals[jj])/num_qubits
                                          ** 2 for ii in range(len(vals)) for jj in range(len(vals)) if np.abs(vals[ii]
                                          + vals[jj]) > 0])
                covar[idx_op2, idx_op1] = covar[idx_op1, idx_op2]


    eigendecomposition = eigen(covar)
    return eigendecomposition[0], eigendecomposition[1]

pts = 5

# passed in params
dissipation_rate = float(sys.argv[1])
model = sys.argv[2]; assert 'simulate' not in model and 'chain' not in model and '_' not in model, f"Only pass in the name of the model Hamiltonian, e.g., {{'ising', 'local_TAT', 'XX'}}"
N = int(sys.argv[3])
coupling_exponent = int(sys.argv[4])
layers = int(sys.argv[5])

obj = getattr(sm, f'simulate_{model}_chain')
obj_params = {'G': sm.collective_op(sm.PAULI_Z, num_qubits=N)/(2*N), 'N': N, 'dissipation':dissipation_rate,
              'coupling_exponent': coupling_exponent}

def min_func(x:np.ndarray, obj_params:dict, layers:int)-> float:

    # hardcode the final two rotations out
    pass_params = np.array([x[ii] if ii <= 2 * layers else 0 for ii in range(3 + 2 * layers)])

    # simulate
    rho = obj(params=pass_params, num_qubits=obj_params['N'], dissipation_rates=obj_params['dissipation'],
              coupling_exponent=obj_params['coupling_exponent'])

    vals, vecs = calc_qfi.compute_eigendecomposition(rho)

    # construct and diagonalize QFIM
    qfi_eigvals, qfi_eigvecs = compute_QFIM(vals, vecs, obj_params['N'])
    return -np.max(qfi_eigvals)

# set up initial vector, parameter bounds
x0, bounds = np.ones(1 + 2 * layers) * 1/2, [(0, 1/2) for _ in range(1 + 2 * layers)]

# seed
random.seed((time.time() * 10**7) % 10**7)
x = x0 * np.random.rand(1 + 2 * layers)

# optimize
out = mini(min_func, x, args=(obj_params, layers, ), tol=1e-2, bounds=bounds, method='Nelder-Mead')


# output
starting_seed = x.tolist()
starting_seed += 2 * [0]
output_params = out.x.tolist()
output_params += 2 * [0]
print(f'Dissipation_rate = {dissipation_rate:.2f}',
      f'Starting seed = {[round(elem, 2) for elem in starting_seed]}',
      f'Output params = {[round(elem, 2) for elem in output_params]}',
      f'Output QFI = {-out.fun}',
      f'**Note that final rotations have been set to zero.**',
      sep='\n')
