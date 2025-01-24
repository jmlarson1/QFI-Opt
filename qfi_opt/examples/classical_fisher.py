import numpy as np
from scipy.optimize import minimize as mini
from scipy.integrate import solve_ivp as ivp
import qfi_opt.spin_models as sm

PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = 1j * PAULI_X @ PAULI_Z
IDENTITY = np.eye(2)


def state_integrator(rho: np.ndarray,
                     Hamiltonian: np.ndarray,
                     phi: float,
                     tol=1e-7) -> np.ndarray:
    '''
    rotator is set to rotate forward, i.e., e^{-i\phi H}
    :param rho: state to rotate
    :param rotator_Hamiltonian: pick a Hamiltonian to rotate with
    :param phi: angle to rotate by
    :param tol: solve_ivp tolerance
    :return:
    '''

    def rotator_function(t, rho, Hamiltonian):
        rho = rho.reshape(len(Hamiltonian), len(Hamiltonian))
        return -1j * (Hamiltonian @ rho - rho @ Hamiltonian).flatten()

    if np.abs(phi) == 0:
        return rho
    else:
        out = ivp(rotator_function, t_span=(0, phi), y0=rho.flatten(), t_eval=np.array([phi]),
                  args=(Hamiltonian,), rtol=tol, atol=tol)

        return (out.y[:, 0]).reshape(len(Hamiltonian), len(Hamiltonian))


def distribution(rho: np.ndarray,
                 num_qubits: int) -> dict:
    assert not np.all(np.isnan(rho.real)), 'density matrix format invalid'

    # build Sz operator and blank dictionary for distribution storage
    Sz = sm.collective_op(sm.PAULI_Z, num_qubits).real
    hist0 = {}

    # iterate over different diagonal values of Sz and populate dictionary
    for k in range(len(Sz)):
        # dictionary item: {Sz matrix element value, [populate only matrix element in question, distribution count, overlap/probability]}
        hist0.setdefault(int(Sz[k, k]), [np.zeros(shape=Sz.shape), 0, 0])
        hist0[int(Sz[k, k])][0][k, k] = 1
        hist0[int(Sz[k, k])][1] += 1

    # calculate the overlap of rho with the independent Sz projections
    for key in hist0:
        hist0[key][2] = float((hist0[key][0] @ rho).trace().real)

    prob_sum = 0
    new_hist = {}
    # populate a prob distribution with the mz projections and the probability of each
    for key in hist0:
        prob_sum += hist0[key][2]
        new_hist.setdefault(key, np.abs(hist0[key][2]))

    # probability sum must be 1.0
    assert np.isclose(prob_sum, 1, 1e-3), "Probability sum is not 1; it is %s." % prob_sum

    # return the probability set / histogram
    return new_hist


def construct_qubit_equator_rotator(num_qubits: int,
                                    which_qubit: int,
                                    phase: float) -> np.ndarray:
    single_rotator = (PAULI_X * np.cos(phase) + PAULI_Y * np.sin(phase)) / 2

    if which_qubit == 0:
        full_rotator = single_rotator
    else:
        full_rotator = IDENTITY

    for idx in range(1, num_qubits):
        if which_qubit == idx:

            full_rotator = np.kron(full_rotator, single_rotator)
        else:
            full_rotator = np.kron(full_rotator, IDENTITY)

    return full_rotator


def construct_qubit_operator(pauli_operator: np.ndarray,
                             num_qubits: int,
                             which_qubit: int) -> np.ndarray:
    if which_qubit == 0:
        single_qubit_operator = pauli_operator / 2
    else:
        single_qubit_operator = IDENTITY

    for idx in range(1, num_qubits):
        if which_qubit == idx:
            single_qubit_operator = np.kron(single_qubit_operator, pauli_operator / 2)
        else:
            single_qubit_operator = np.kron(single_qubit_operator, IDENTITY)

    return single_qubit_operator


def compute_collective_CFI(distribution1: dict,
                           distribution2: dict,
                           phi: float) -> float:
    sum = 0
    for key in distribution1:
        sum += (np.sqrt(distribution1[key]) - np.sqrt(distribution2[key])) ** 2

    return 4 * sum / phi ** 2

# brute force find the best axis for a rotation uniformly applied to all qubits
def compute_collective_basis_CFI_for_uniform_qubit_rotations(rho: np.ndarray,
                   num_qubits: int,
                   varphi_partition: int = 31) -> [float, float]:

    assert not np.all(np.isnan(rho.real)), 'density matrix format invalid'

    dphi = 1e-5
    Sx, Sy, Sz = sm.collective_spin_ops(num_qubits=num_qubits)

    def Svarphi(varphi: float) -> np.ndarray:
        return np.cos(varphi) * Sx + np.sin(varphi) * Sy

    varphi = np.linspace(0, np.pi / 2, varphi_partition)
    fishers = np.zeros(varphi_partition)

    # brute force search optimal axis
    for jj in range(varphi_partition):
        Svarphi_ = Svarphi(varphi[jj])
        rho_varphi = state_integrator(rho, Svarphi_, np.pi / 2)
        rho_pert = state_integrator(rho, Sz, dphi)
        rho_varphi_pert = state_integrator(rho_pert, Svarphi_, np.pi / 2)

        unpert_dist = distribution(rho_varphi, num_qubits)
        pert_dist = distribution(rho_varphi_pert, num_qubits)
        fishers[jj] = compute_collective_CFI(unpert_dist, pert_dist, dphi)

    # NOTE: CFI IS NORMALIZED TO HEISENBERG LIMIT
    # return CFI and optimal rotation axes
    return np.max(fishers) / num_qubits ** 2, varphi[np.argmax(fishers)]


def compute_collective_basis_CFI_for_uniform_qubit_rotations_Ffun(params, sim_params):

    num_params = len(params)
    x = params[:num_params - 1]
    theta = params[num_params - 1] # theta is one dimensional for this CFI type.

    num_qubits = sim_params['N']
    model = sim_params['model']
    coupling_exponent = sim_params['coupling_exponent']
    dissipation_rates = sim_params['dissipation_rates']
    dphi = sim_params['dphi'] # dphi = 1e-5

    simulation_obj = getattr(sm, f'simulate_{model}_chain')

    rho = simulation_obj(params=x, num_qubits=num_qubits, dissipation_rates=dissipation_rates,
                         coupling_exponent=coupling_exponent)

    assert not np.all(np.isnan(rho.real)), 'density matrix format invalid'

    Sx, Sy, Sz = sm.collective_spin_ops(num_qubits=num_qubits)

    def Svarphi(varphi: float) -> np.ndarray:
        return np.cos(varphi) * Sx + np.sin(varphi) * Sy

    Svarphi_ = Svarphi(theta)
    rho_varphi = state_integrator(rho, Svarphi_, np.pi / 2)
    rho_pert = state_integrator(rho, Sz, dphi)
    rho_varphi_pert = state_integrator(rho_pert, Svarphi_, np.pi / 2)

    unpert_dist = distribution(rho_varphi, num_qubits)
    pert_dist = distribution(rho_varphi_pert, num_qubits)

    # note: h needs to be multiplied by 1.0 / ((num_qubits * dphi) ** 2)
    return np.concatenate((unpert_dist, pert_dist))


def compute_bitstring_basis_CFI_for_uniform_qubit_rotations(rho: np.ndarray,
                   num_qubits: int,
                   varphi_partition: int = 31) -> [float, float]:
    assert not np.all(np.isnan(rho.real)), 'density matrix format invalid'

    dphi = 1e-5
    Sx, Sy, Sz = sm.collective_spin_ops(num_qubits=num_qubits)

    def Svarphi(varphi: float) -> np.ndarray:
        return np.cos(varphi) * Sx + np.sin(varphi) * Sy

    varphi = np.linspace(0, np.pi / 2, varphi_partition)
    fishers = np.zeros(varphi_partition)

    # brute force search optimal rotation axis
    for jj in range(varphi_partition):
        Svarphi_ = Svarphi(varphi[jj])
        rho_varphi = state_integrator(rho, Svarphi_, np.pi / 2)

        # rotate about z then about the rotation axis set by Svarphi
        rho_pert = state_integrator(rho, Sz, dphi)
        rho_varphi_pert = state_integrator(rho_pert, Svarphi_, np.pi / 2)

        pert_dist = np.diagonal(rho_varphi_pert)
        init_dist = np.diagonal(rho_varphi)
        fishers[jj] = 4 / dphi ** 2 * np.sum((np.sqrt(pert_dist) - np.sqrt(init_dist)) ** 2)

    # NOTE: CFI IS NORMALIZED TO HEISENBERG LIMIT
    # return CFI and optimal rotation axes
    return np.max(fishers) / num_qubits ** 2, varphi[np.argmax(fishers)]



# we instead optimize to find the best rotation axes for all single-qubit rotations
def compute_collective_basis_CFI_for_single_qubit_rotations(rho: np.ndarray,
                                                      num_qubits: int) -> list:
    Sx, Sy, Sz = sm.collective_spin_ops(num_qubits)

    # define function to minimize
    def min_funct(x0: np.ndarray, init_rho: np.ndarray, num_qubits: int):
        dphi = 1e-5

        # rotate all qubits uniformly by small value dphi about z
        rotate_qubit = state_integrator(init_rho, Sz, dphi)

        # rotate all qubits about the equator arbitrarily
        for qubit_idx in range(num_qubits):
            rotator = construct_qubit_equator_rotator(num_qubits, qubit_idx, float(x0[qubit_idx]))
            rotate_qubit = state_integrator(rotate_qubit, rotator, np.pi / 2)
            init_rho = state_integrator(init_rho, rotator, np.pi / 2)

        # compute the cfi with respect to the arbitrarily rotated density matrix
        rotated_rho_dist = distribution(rotate_qubit, num_qubits)
        init_rho_dist = distribution(init_rho, num_qubits)

        CFI = -compute_collective_CFI(init_rho_dist, rotated_rho_dist, dphi)
        return CFI

    opt_cfi = 0
    attempts = 5  # // num_qubits
    bnds = [(0, np.pi) for _ in range(num_qubits)]
    for attempt in range(attempts):
        x0 = np.ones(num_qubits) * np.pi * np.random.rand(num_qubits)
        out = mini(min_funct, x0=x0, method='Nelder-Mead', tol=1e-2, args=(rho, num_qubits,), bounds=bnds)

        if out.fun < opt_cfi:
            opt_cfi = out.fun
            opt_x = out.x

    # NOTE: CFI IS NORMALIZED TO HEISENBERG LIMIT
    # return CFI and optimal single-qubit rotation axes
    return -opt_cfi/num_qubits**2, opt_x


def compute_bitstring_basis_CFI_for_single_qubit_rotations(rho: np.ndarray,
                                                     num_qubits: int) -> list:
    Sx, Sy, Sz = sm.collective_spin_ops(num_qubits)

    # define function to minimize
    def min_funct(x0: np.ndarray, init_rho: np.ndarray, num_qubits: int) -> float:
        dphi = 1e-5

        # rotate all qubits uniformly by small z value
        rotate_qubit = state_integrator(init_rho, Sz, dphi)

        # rotate all qubits about the equator about axis given by x0
        for qubit_idx in range(num_qubits):
            rotator = construct_qubit_equator_rotator(num_qubits, qubit_idx, float(x0[qubit_idx]))
            rotate_qubit = state_integrator(rotate_qubit, rotator, np.pi / 2)
            init_rho = state_integrator(init_rho, rotator, np.pi / 2)

        # compute the cfi with respect to the probabilities of the rotated/original distributions
        # with same single-qubit rotations
        pert_dist = np.diagonal(rotate_qubit)
        init_dist = np.diagonal(init_rho)
        CFI = -4 / dphi ** 2 * np.sum((np.sqrt(pert_dist) - np.sqrt(init_dist)) ** 2)

        return CFI

    # maximize CFI by optimizing over all single-qubit rotation axes, varphi_j
    opt_cfi = 0
    attempts = 10  # 50 // num_qubits
    bnds = [(0, np.pi) for _ in range(num_qubits)]
    for attempt in range(attempts):

        x0 = np.ones(num_qubits) * np.pi * np.random.rand(num_qubits)
        out = mini(min_funct, x0=x0, method='Nelder-Mead', tol=1e-2, args=(rho, num_qubits,), bounds=bnds)

        if out.fun < opt_cfi:
            opt_cfi = out.fun
            opt_x = out.x

    # NOTE: CFI IS NORMALIZED TO HEISENBERG LIMIT
    # return CFI and optimal single-qubit rotation axes
    return -opt_cfi/num_qubits**2, opt_x






