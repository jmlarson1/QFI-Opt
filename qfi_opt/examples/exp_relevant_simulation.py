"""
Simulate QFI-Opt by dynamically swapping between qfi_opt.examples.PermSolver when coupling_exponent = 0 or
qfi_opt.spin_models when coupling_exponent > 0. The collective models OAT and TAT or Ising, XX and
local_TAT may be simulated with respective values of coupling_exponent.

# system sizes
Typical system sizes of num_qubits <= 10 are experimentally feasible. Special cases involving the collective treatment
(i.e. coupling_exponent = 0) of spin ensembles (for example, trapped-ions in a Penning trap) may reach system sizes on
the order of hundreds of ions.

# interaction ranges
Typical experimentally relevant interaction range test cases are (i) all-to-all (i.e. infinite-range interaction)
pertaining to coupling_exponent = 0, (ii) intermediate-range interactions with coupling_exponent = 3 (e.g., dipole-dipole)
and (iii) interaction ranges pushing into the nearest-neighbor regime (e.g., van der Waals interactions) with
coupling_exponent \geq 6.

# noise
It is difficult to set specific noise values to test as these tend to be experimentally specific. However, we can probe
some noise rates that may otherwise be relevant to state-of-the-art experiments. Noise rate scales that may be of
physical interset are: (i) the ideal case in which dissipation_rates = 0 and with which analytic
comparisons can be made against, (ii) the vanishingly weak case with dissipation_rates \approx 0.01 pertaining to cases
that are near-ideal and are subject to perturbative noise, (iii) increasing orders of magnitude beyond the perturbative
regime but less than order 1 as our results suggest that noise rates beyond this upper limit render states with
metrological potential below the classical limit.
"""
import sys, os

import qfi_opt.spin_models as sm
from qfi_opt.examples import calculate_qfi as calc_qfi
from qfi_opt.examples import PermSolver_methods as methods
from qfi_opt.examples import PermSolver_matrix as matrix

import functools as ftools
import warnings
import numpy as np

""" 
# raw inputs are all strings
# dissipation_rates: pass one float (if all dissipation rates are equal), or a string of three floats separated by ","
# params: pass a string of numbers separated by "," of length 3 + 2 * l where l is the number of layers
"""
num_qubits = int(sys.argv[1])
model = sys.argv[2]
coupling_exponent = float(sys.argv[3])
dissipation_rates = sys.argv[4]
dissipation_rates = float(dissipation_rates) if sys.argv[4].count(',') == 0 else tuple(float(rate) for rate in dissipation_rates.split(','))
layers = int(sys.argv[5])

#params = [float(param) for param in sys.argv[5].split(',')]

assert model in ['ising', 'local_TAT', 'XX'], f'Passed model "{model}" not in expected models: "ising", "local_TAT", "XX"'
assert coupling_exponent >= 0, f'Coupling exponent {coupling_exponent} is unphysical.'

params = (np.array([1/2 for _ in range(2)] + [1/2 if _ % 2 else 1 for _ in range(2 * layers)] + [1])
      * np.random.rand(3 + 2 * layers))

assert len(params) >= 5 and (len(params) - 3) % 2 == 0, f'Unexpected param length {len(params)}. Should follow 3 + 2 * l where l is the number of layers.'



if (model == 'XX' and coupling_exponent == 0):
    warnings.warn("When coupling_exponent = 0, 'XX' and 'ising' are identical up to a phase (a factor of -1) which may affect the geometry of "
                  "entanglement generation and state sensitivity to certain decoherence.")

obj_params = {"N":num_qubits, "params":params, "dissipation":dissipation_rates}
if coupling_exponent == 0:
    # map short-range to infinite-range
    parent_models = {"ising":"OAT", "XX":"OAT", "local_TAT":"TAT"}

    # construct Hamiltonian for infinite-range model
    Hamiltonian_set = getattr(matrix, f'{parent_models[model]}Mat')(1.0, num_qubits//2)

    # simulate
    rho = methods.simulate_layers(params=params, num_qubits=num_qubits, Hamiltonian_set=Hamiltonian_set, dissipation_rates=dissipation_rates)

    # jacobian
    get_jacobian = methods.get_jacobian_func(methods.simulate_layers)
    jacobian = ftools.partial(get_jacobian,
                                  num_qubits=num_qubits,
                                  Hamiltonian_set=Hamiltonian_set,
                                  dissipation_rates=dissipation_rates)

    # compute QFI
    obj_params["G"] = getattr(methods, 'MatSz')(num_qubits // 2)
    ps_qfi, grad = methods.compute_QFI(rho=rho, params=params, jacobian=jacobian, obj_params=obj_params)
    print(f'PermSolver: \nQFI: {ps_qfi} \nparams: {[float(f"{_:.3f}") for _ in params]}')


else:
    # simulate
    sim_obj = getattr(sm, f'simulate_{model}_chain')
    rho = sim_obj(params=params, num_qubits=num_qubits, dissipation_rates=dissipation_rates, coupling_exponent=coupling_exponent)

    # jacobian
    # TODO: need to pass coupling_exponent?
    get_jacobian = sm.get_jacobian_func(sim_obj)
    #jacobian = ftools.partial(get_jacobian,
    #                              num_qubits=num_qubits,
    #                              dissipation_rates=dissipation_rates,
    #                              coupling_exponent=coupling_exponent)
    grad = np.zeros(len(params))

    # compute QFI
    vals, vecs = calc_qfi.compute_eigendecomposition(rho)
    obj_params["G"] = sm.collective_op(sm.PAULI_Z, num_qubits) / 2
    sm_qfi, grad = calc_qfi.compute_QFI(rho=rho, eigvals=vals, eigvecs=vecs, params=params, obj_params=obj_params, get_jacobian=get_jacobian, grad=grad)
    print(f'spin_models: \nQFI: {sm_qfi/num_qubits**2} \nparams: {[float(f"{_:.3f}") for _ in params]}')
