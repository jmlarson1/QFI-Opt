import sys

import os
USE_DIFFRAX = bool(os.getenv("USE_DIFFRAX"))

if USE_DIFFRAX:
    import diffrax
    import jax
    import jax.numpy as np
    from jax.scipy.linalg import expm
    from jax.numpy import linalg as LA
    jax.config.update("jax_enable_x64", True)

else:
    import numpy as np  # type: ignore[no-redef]
    from scipy.linalg import expm
    from numpy import linalg as LA

import PermSolver_matrix as matrix
import PermSolver_methods as methods


N = int(sys.argv[1])
model = sys.argv[2]
assert model in ['OAT', 'TAT'], "passed model not OAT or TAT"
dissipation = float(sys.argv[3])
layers = int(sys.argv[4])
calcjacobian = int(sys.argv[5])

Jmax = N//2
G = methods.MatSz(Jmax)
exec(f"Hmat_set = matrix.{model}Mat(1.0, Jmax)")

# params
if USE_DIFFRAX == False:
      x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * layers)] + [1])
            * np.random.rand(3 + 2 * layers))
else:
      x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * layers)] + [1])
            * np.random.rand(3 + 2 * layers))

# simulate
if calcjacobian == 0:
      rho = methods.simulate_layers(params=x0,
                                     num_qubits=N,
                                     Hamiltonian_set=Hmat_set,
                                     dissipation_rates=dissipation / np.pi)
      # calculate qfi
      qfi = methods.calc_QFI(rho, Jmax, G) / N**2
      print(f'params = {x0}', f'qfi = {qfi}', sep='\n')
else:
      get_jacobian = methods.get_jacobian_func(methods.simulate_layers)
      jacobian = get_jacobian(params=x0,
                                     num_qubits=N,
                                     Hamiltonian_set=Hmat_set,
                                     dissipation_rates=dissipation / np.pi)
      for i in range(len(jacobian)):
            if USE_DIFFRAX == False:
                  methods.print_jacobian_manual(jacobian[i])
            else:
                  methods.print_jacobian(jacobian[i])



