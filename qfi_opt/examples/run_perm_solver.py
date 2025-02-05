import numpy as np
import sys

import PermSolver_matrix as matrix
import PermSolver_methods as methods


N = int(sys.argv[1])
model = sys.argv[2]
assert model in ['OAT', 'TAT'], "passed model not OAT or TAT"
dissipation = float(sys.argv[3])
layers = int(sys.argv[4])

Jmax = N//2
G = methods.MatSz(Jmax)
exec(f"Hmat_set = matrix.{model}Mat(1.0, Jmax)")

# params
x0 = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * layers)] + [1])
      * np.random.rand(3 + 2 * layers))

# simulate
rho = methods.simulate_layers(params=x0,
                                     num_qubits=N,
                                     Hamiltonian_set=Hmat_set,
                                     dissipation_rates=dissipation / np.pi)

# calculate qfi
qfi = methods.calc_QFI(rho, Jmax, G) / N**2

print(f'params = {x0}', f'qfi = {qfi}', sep='\n')


