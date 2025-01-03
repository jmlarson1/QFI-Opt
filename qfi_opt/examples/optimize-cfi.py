import numpy as np
import classical_fisher as cf
import qfi_opt.spin_models as sm
from scipy.optimize import minimize as mini
import time


# Set up
########################################################################################################################
# choose one of the cfi computations in the following list
# for reference, cfi_types[0] is the computation used to maximize Eq. 10 in our recently published paper
cfi_types = ['compute_collective_basis_CFI_for_uniform_qubit_rotations',
            'compute_bitstring_basis_CFI_for_uniform_qubit_rotations',
            'compute_collective_basis_CFI_for_single_qubit_rotations',
            'compute_bitstring_basis_CFI_for_single_qubit_rotations']


# finds cfi for given rho and num_qubits
compute_cfi = getattr(cf, cfi_types[0])

# simulation parameters
N = 4
model = 'XX'
coupling_exponent = 3
dissipation_rates = 1
layers = 1
simulation_obj = getattr(sm, f'simulate_{model}_chain')

# parameter bounds and maximum input params
bounds = [(0, 1/2), (0, 1/2)] + [(0, 1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)]
input_params = np.array(2 * [1/2] + [1/2 if _ % 2 == 0 else 1 for _ in range(2 * layers)] + [1])

# Optimize: find simulation parameters that give rho with best CFI
########################################################################################################################
def min_funct(x0: np.ndarray):
    # simulate
    rho = simulation_obj(params=x0,
                         num_qubits=N, dissipation_rates=dissipation_rates, coupling_exponent=coupling_exponent)

    # compute cfi for rho with given parameters
    cfi, _ = compute_cfi(rho, N)

    return -cfi

optimization_attempts = 2
opt_cfi = 0
for attempt in range(optimization_attempts):

    np.random.seed(int(time.time() * 10**7) % 10**6)
    params = input_params * np.random.rand(len(input_params))

    out = mini(min_funct, x0=params, method='Nelder-Mead', bounds=bounds, tol=1e-2)

    print(f'returned CFI = {-out.fun:.2f}',
          f'returned params = {out.x.tolist()}', sep='\n')

    if out.fun < opt_cfi:
        opt_cfi = out.fun
        opt_params = out.x

    print(f'Current Opt. CFI = {-opt_cfi:.2f}\n')


print(f'\nOptimal CFI = {-opt_cfi:.2f}',
      f'Optimal params = {opt_params.tolist()}', sep='\n')
