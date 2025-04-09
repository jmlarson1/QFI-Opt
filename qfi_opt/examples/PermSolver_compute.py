import os, sys, time
import numpy as np

import PermSolver_matrix as matrix
import PermSolver_methods as methods

import scipy
from scipy.integrate import solve_ivp as ivp
from scipy.optimize import minimize as mini
from scipy.special import binom

import matplotlib
import matplotlib.pyplot as plt

import functools


def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues, eigenVectors

def init_rho(num_qubits:int)->list:
    Jmax = int(np.ceil(num_qubits / 2))

    rho0 = []
    product_down = eigen(construct_Jz(num_qubits)[-1])[1][:, 0]
    product_down = product_down.reshape((len(product_down), 1))
    for Jth_sector in range(Jmax):
        rho0.append(np.zeros((2 * Jth_sector + 1, 2 * Jth_sector + 1), dtype='complex128'))

    rho0.append(np.kron(product_down.T.conj(), product_down))

    return rho0

def construct_identity(num_qubits:int)->list:
    '''
    :param num_qubits: number of system qubits
    :return: the identity operator across all J sectors spanning J = {N/2, N/2 - 1, . . ., 0}
    '''
    # TODO: FIX CONSTRUCTORS TO WORK WITH ODD N
    assert num_qubits % 2 == 0, 'Odd N does not work at the moment.'

    Jmax = int(np.ceil(num_qubits / 2))
    identity = []
    for Jth_sector in range(Jmax + 1):
        Jth_sector_val = Jth_sector if num_qubits % 2 == 0 else Jth_sector / 2
        Jth_matrix = np.identity(int(2 * Jth_sector_val + 1), dtype='complex128')
        identity.append(Jth_matrix)

    return identity

def construct_Jz(num_qubits:int)->list:
    '''
    :param num_qubits: number of system qubits
    :return: the collective Jz operator across all J sectors spanning J = {N/2, N/2 - 1, . . ., 0}
    '''
    # TODO: FIX CONSTRUCTORS TO WORK WITH ODD N
    assert num_qubits % 2 == 0, 'Odd N does not work at the moment.'

    Jmax = int(np.ceil(num_qubits / 2))
    Jz = []
    for Jth_sector in range(Jmax + 1):
        Jth_sector_val = Jth_sector if num_qubits % 2 == 0 else Jth_sector / 2
        # print(Jth_sector, int(Jth_sector_val), int(Jth_sector_val * 2 + 1), sep=', ')
        Jth_matrix = np.zeros(shape=(int(2 * Jth_sector_val + 1), int(2 * Jth_sector_val + 1)), dtype='complex128')

        for mj_idx, mj in enumerate(np.arange(Jth_sector_val, -Jth_sector_val - 1, -1)):
            Jth_matrix[mj_idx, mj_idx] = mj

        Jz.append(Jth_matrix)

    return Jz

def construct_Jp(num_qubits:int)->list:
    '''
    :param num_qubits: number of system qubits
    :return: the collective spin raising operator across all J sectors spanning J = {N/2, N/2 - 1, . . ., 0}
    '''

    # TODO: FIX CONSTRUCTORS TO WORK WITH ODD N
    assert num_qubits % 2 == 0, 'Odd N does not work at the moment.'

    Jmax = int(np.ceil(num_qubits / 2))
    Jp = []
    for Jth_sector in range(Jmax + 1):
        Jth_sector_val = Jth_sector if num_qubits % 2 == 0 else Jth_sector / 2
        Jth_matrix = np.zeros(shape=(int(2 * Jth_sector_val + 1), int(2 * Jth_sector_val + 1)), dtype='complex128')

        for mj_idx, mj in enumerate(np.arange(Jth_sector_val - 1, -Jth_sector_val - 1, -1)):
            Jth_matrix[mj_idx, mj_idx + 1] = np.sqrt((Jth_sector_val - mj) * (Jth_sector_val + mj + 1))

        Jp.append(Jth_matrix)

    return Jp

def construct_Jm(num_qubits:int, constructed_Jp:list=[])->list:
    '''
    :param num_qubits: number of system qubits
    :param constructed_Jp: if constructed Jp has already been constructed, pass it to avoid reconstruction
    :return: the collective spin lowering operator across all J sectors spanning J = {N/2, N/2 - 1, . . ., 0}
    '''

    if not constructed_Jp:
        constructed_Jp = construct_Jp(num_qubits)

    return [Jp_transpose.T for Jp_transpose in constructed_Jp]

def construct_Jx(num_qubits:int)->list:
    '''
    Use the raising and lowering operators to construct Jx.
    :param num_qubits: number of system qubits
    :return:
    '''
    Jp = construct_Jp(num_qubits)
    Jm = construct_Jm(num_qubits, constructed_Jp=Jp)

    return [(Jp[Jth_sector] + Jm[Jth_sector]) / 2 for Jth_sector in range(len(Jp))]

def construct_Jy(num_qubits:int)->list:
    '''
    Use the raising and lowering operators to construct Jy.
    :param num_qubits: number of system qubits
    :return: Jy across all J sectors spanning J = {N/2, N/2 - 1, . . ., 0}
    '''
    Jp = construct_Jp(num_qubits)
    Jm = construct_Jm(num_qubits, constructed_Jp=Jp)

    return [(Jp[Jth_sector] - Jm[Jth_sector]) / (2j) for Jth_sector in range(len(Jp))]

def expectation(rho, num_qubits:int, operator):
    Jmax = int(np.ceil(num_qubits/2))
    return sum(np.trace(rho[Jth_sector] @ operator[Jth_sector]) for Jth_sector in range(Jmax + 1))

def covariance(rho, num_qubits:int, operator1, operator2):
    # compute < O1 O2 > - < O1 > < O2 >
    conjoined_operator = matrix_multiply_across_sectors(operator1, operator2, num_qubits)
    return expectation(rho, num_qubits, conjoined_operator) - expectation(rho, num_qubits, operator1) * expectation(rho, num_qubits, operator2)

def variance(rho:list, num_qubits:int, operator):
    return covariance(rho, num_qubits, operator, operator)

def collective_projection(rho, num_qubits:int, operator_moments:list=[]):
    if operator_moments == []:
        operator_moments = construct_spin_mats(num_qubits)

    return sum(np.abs(expectation(rho, num_qubits, operator))**2 for operator in operator_moments)

def polarization(rho:list, num_qubits:int, theta:float, phi:float, operator_moments:list=[])->float:
    if operator_moments == []:
        operator_moments = construct_spin_mats(num_qubits)

    coefficients = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    polarization_set = [matrix_multiply_across_sectors(operator_moment, coefficient, num_qubits) for operator_moment, coefficient in zip(operator_moments, coefficients)]
    polarization_operator = matrix_sum_across_sectors(matrix_sum_across_sectors(polarization_set[0], polarization_set[1], num_qubits), polarization_set[2], num_qubits)
    return expectation(rho, num_qubits, polarization_operator)

def find_state_polarization(rho:list, num_qubits:int, max_attempts:int=20)->[float, float]:
    def min_funct(x0:np.ndarray, rho, num_qubits:int, operator_moments:list)->float:
        return -polarization(rho, num_qubits, x0[0], x0[1], operator_moments)

    operator_moments = construct_spin_mats(num_qubits)
    max_polarization = 0
    opt_polarization_angles = np.zeros(2)
    input_params = np.array([np.pi, 2 * np.pi])
    for attempt in range(max_attempts):
        np.random.seed(int((time.time() * 10 ** 7) % 10**7))
        result = mini(min_funct, input_params * np.random.rand(2), args=(rho, num_qubits, operator_moments), tol=1e-2, method='Nelder-Mead', bounds=[(0, np.pi), (0, np.pi * 2)])

        if result.fun < max_polarization:
            max_polarization = result.fun
            opt_polarization_angles = result.x

    return opt_polarization_angles.tolist()

def polarize_state_along_x(rho:list, num_qubits:int):
    # TODO: apply two rotations instead of one
    theta, phi = find_state_polarization(rho, num_qubits)
    Jz = construct_Jz(num_qubits)
    return state_integrator(rho, num_qubits, Jz, -phi)

def construct_spin_mats(num_qubits:int, return_all:bool=False, return_raising_and_lowering:bool = False, return_identity:bool = False)->list:

    Jx = construct_Jx(num_qubits)
    Jy = construct_Jy(num_qubits)
    Jz = construct_Jz(num_qubits)

    if return_all:
        Jp = construct_Jp(num_qubits)
        return Jx, Jy, Jz, Jp, construct_Jm(num_qubits, Jp), construct_identity(num_qubits)

    elif return_identity and not return_raising_and_lowering:
        return Jx, Jy, Jz, construct_identity(num_qubits)

    elif not return_identity and return_raising_and_lowering:
        Jp = construct_Jp(num_qubits)
        return Jx, Jy, Jz, Jp, construct_Jm(num_qubits, Jp)

    else:
        return Jx, Jy, Jz

def construct_GHZ(num_qubits:int, phase:float)->np.ndarray:
    '''
    construct a GHZ state with given phase in the z basis
    :param num_qubits: number of system qubits
    :param phase: dictates linear combination of all spin up plus all spin down
    :return: GHZ state in largest Jth sector
    '''
    Jz = construct_Jz(num_qubits)
    z_vals, z_vecs = eigen(Jz[-1])
    all_down, all_up = z_vecs[:,0].reshape(len(z_vecs[:,0]), 1), z_vecs[:, -1].reshape(len(z_vecs[:,0]), 1)
    GHZ = (all_up + np.exp(1j * phase) * all_down) / np.sqrt(2)
    return np.kron(GHZ.T.conj(), GHZ)

def collectivity2(rho:list, num_qubits:int)->list:
    Jmax = int(np.ceil(num_qubits / 2))
    operators = construct_spin_mats(num_qubits)
    collective_operator =  [matrix_multiply_across_sectors(operator, operator, num_qubits) for operator in operators]
    collective_operator = matrix_sum_across_sectors(matrix_sum_across_sectors(*collective_operator[:2], num_qubits), collective_operator[-1], num_qubits)
    collectivity_by_sector = []
    for Jth_sector in range(Jmax + 1):
        collectivity_by_sector.append((rho[Jth_sector] @ collective_operator[Jth_sector]).trace() / (Jth_sector * (Jth_sector + 1)))

    return collectivity_by_sector

def matrix_multiply_across_sectors(mat1:list, mat2:list, num_qubits:int)->list:
    Jmax = int(np.ceil(num_qubits / 2))

    if type(mat2) == list:
        return [mat1[sector] @ mat2[sector] for sector in range(Jmax + 1)]
    elif type(mat2) in [int, float, np.float64]:
        return [mat1[sector] * mat2 for sector in range(Jmax + 1)]

def matrix_sum_across_sectors(mat1:list, mat2:list, num_qubits:int)->list:
    Jmax = int(np.ceil(num_qubits / 2))
    return [mat1[sector] + mat2[sector] for sector in range(Jmax + 1)]

def state_integrator(rho: list,
                     num_qubits: int,
                     Hamiltonian: list,
                     t: float,
                     tol=1e-10) -> np.ndarray:
    '''
    rotator is set to evolve forward, i.e., e^{-i H t}
    :param rho: state to evolve
    :param Hamiltonian: pick a Hamiltonian to evolve with
    :param t: time to evolve by
    :param tol: solve_ivp tolerance
    :return:
    '''

    # print(*[rhoJ.shape for rhoJ in rho])
    Jmax = int(np.ceil(num_qubits / 2))
    evolved_state = []

    def evolve_function(t, Jth_rho, Jth_Hamiltonian):
        Jth_rho = Jth_rho.reshape(len(Jth_Hamiltonian), len(Jth_Hamiltonian))
        return -1j * (Jth_Hamiltonian @ Jth_rho - Jth_rho @ Jth_Hamiltonian).flatten()

    if np.abs(t) == 0:
        return rho

    else:
        for Jth_sector in range(Jmax + 1):
            out = ivp(evolve_function, t_span=(0, t), y0=rho[Jth_sector].flatten(), t_eval=np.array([t]),
                      args=(Hamiltonian[Jth_sector], ), rtol=tol, atol=tol)

            evolved_state.append((out.y[:, 0]).reshape(len(Hamiltonian[Jth_sector]), len(Hamiltonian[Jth_sector])))

        return evolved_state

def quick_check(rho:list, num_qubits:int)->None:
    operators = construct_spin_mats(num_qubits)
    print(*[f'<S{alpha}>/J = {expectation(rho, num_qubits, operators[alpha_idx]).real / (num_qubits / 2):.2f}' for alpha_idx, alpha in enumerate(['x', 'y', 'z'])], sep='\n')
    print(*[f'<var(S{alpha})>/(J/2) = {variance(rho, num_qubits, operators[alpha_idx]).real / (num_qubits / 4):.2f}' for alpha_idx, alpha in enumerate(['x', 'y', 'z'])], sep='\n')

def histogram(rho:list, num_qubits:int, only_full_histogram:bool=False, save_location:str='', plot_probabilities_greater_than:float=1e-2)->None:
    Jmax = int(np.ceil(num_qubits / 2))
    full_dist = distribution(rho, num_qubits)

    full_histogram = {mz:0 for mz in range(-Jmax, Jmax + 1)}
    for Jth_sector in range(Jmax + 1):
        for mz in range(-Jth_sector, Jth_sector + 1):
            full_histogram[mz] += full_dist[Jth_sector][mz]


        if not only_full_histogram:
            if np.greater(sum(list(full_dist[Jth_sector].values())), plot_probabilities_greater_than):
                plt.figure(layout='tight')
                plt.bar(full_dist[Jth_sector].keys(), full_dist[Jth_sector].values())
                plt.xticks(list(full_dist[Jth_sector].keys()))
                plt.ylabel(r'$P(J, m_z)$', fontsize=plf.fs)
                plt.xlabel(r'$m_z$', fontsize=plf.fs)
                plt.title(rf'$N = {num_qubits}, J = {Jth_sector}, \;$' + '$\sum_{m_z}{P(J, m_z)} = %s$'%f'{sum(list(full_dist[Jth_sector].values())):.2f}', fontsize=plf.fs)
                if np.max(list(full_dist[Jth_sector].values())) >= 0.4:
                    plt.ylim(0, 1.0)
                else:
                    plt.ylim(0, 0.5)

                if save_location:
                    plt.savefig(save_location % f'histograms/N{num_qubits}, J={Jth_sector}.png', dpi=300)
                    plt.show(block=False)
                    plt.close()


    plt.figure(layout='tight')
    plt.bar(full_histogram.keys(), full_histogram.values())
    plt.xticks(list(full_histogram.keys()))
    plt.yticks(np.arange(0, 11)*0.1)
    plt.ylabel(r'$\sum_{J = 0}^{J_{\mathrm{max}}}{P(J, m_z)}$', fontsize=plf.fs)
    plt.xlabel(r'$m_z$', fontsize=plf.fs)
    if np.max(list(full_histogram.values())) >= 0.4:
        plt.ylim(0, 1.0)
    else:
        plt.ylim(0, 0.5)
    plt.title(r'$N = %s, \; \mathrm{All} \; J$'%str(num_qubits), fontsize=plf.fs)
    if save_location:
        plt.savefig(save_location % f'histograms/N{num_qubits}, all J.png', dpi=300)
        plt.show(block=False)
        plt.close()

# relevant quantities
def GHZ_fidelity(rho:list, num_qubits:int, optimization_attempts:int=20)->[float, float]:
    '''
    compute the optimal GHZ fidelity of a state rho by finding the optimal phase
    :param rho: state of interest
    :param num_qubits: number of system qubits
    :param optimization_attempts: number of attempts each beginning from a random starting point
    :return:
    '''
    rho = rho[-1]

    def min_funct(x0:float, rho:np.ndarray, num_qubits:int)->float:
        return -(construct_GHZ(num_qubits, x0) @ rho).trace()

    opt_phase = 0
    best_fidelity = 0
    for attempt in range(optimization_attempts):
        np.random.seed(int(time.time() * 10**7) % 10 ** 7)
        out = mini(min_funct, x0=2 * np.pi * np.random.rand(1), args=(rho, num_qubits, ), method='Nelder-Mead', tol=1e-2, bounds=[(0,  2 * np.pi)])
        if out.fun < best_fidelity:
            best_fidelity = out.fun
            opt_phase = out.x

    return [-best_fidelity, opt_phase]

def classical_Fisher(rho:list, num_qubits:int, total_attempts:int=10)->[float, float]:
    operators = construct_spin_mats(num_qubits)
    def axis_rotator(num_qubits:int, operators:list, phi:float)->list:

        Sx = [sub_op * np.cos(phi) for sub_op in operators[0]]
        Sy = [sub_op * np.sin(phi) for sub_op in operators[1]]
        return matrix_sum_across_sectors(Sx, Sy, num_qubits)

    def minimize_cfi(phi:float, rho:list, num_qubits:int, operators:list)->float:
        # TODO: CLEAN UP
        rotator = axis_rotator(num_qubits, operators, phi)
        dtheta = 1e-7
        rotated_rho = state_integrator(rho=rho, num_qubits=num_qubits, Hamiltonian=rotator, t=np.pi/2)

        rho_pert = state_integrator(rho=rho, num_qubits=num_qubits, Hamiltonian=operators[-1], t=dtheta)
        rotated_rho_pert = state_integrator(rho=rho_pert, num_qubits=num_qubits, Hamiltonian=rotator, t=np.pi/2)

        rho_distribution = distribution(rotated_rho_pert, num_qubits)
        rho_dtheta_distribution = distribution(rotated_rho, num_qubits)

        Jmax = int(np.ceil(num_qubits / 2))
        rho_mz = {mz: 0 for mz in range(-Jmax, Jmax + 1)}
        rho_dtheta_mz = {mz: 0 for mz in range(-Jmax, Jmax + 1)}
        for Jth_sector in range(Jmax + 1):
            for mz in range(-Jth_sector, Jth_sector + 1):
                rho_mz[mz] += rho_distribution[Jth_sector][mz]
                rho_dtheta_mz[mz] += rho_dtheta_distribution[Jth_sector][mz]

        running_cfi_sum = 0
        for mz in range(-Jmax, Jmax + 1):
            running_cfi_sum += (np.sqrt(rho_mz[mz]) - np.sqrt(rho_dtheta_mz[mz]))**2

        return -4 * running_cfi_sum / dtheta**2

    optimal_cfi, optimal_rot_axis = 0, 0
    for attempt in range(total_attempts):
        out = mini(minimize_cfi, x0=np.pi * np.random.rand(1), args=(rho, num_qubits, operators,),
                   method='Nelder-Mead', tol=1e-2, bounds=[(0, np.pi)])


        if out.fun < optimal_cfi:
            optimal_cfi = out.fun
            optimal_rot_axis = out.x

    return -optimal_cfi, optimal_rot_axis

def squeezing(rho:list, num_qubits:int, optimization_attempts:int=20)->[float, float, float]:
    def minimize_polarization(x0:np.ndarray, rho, num_qubits:int, Sx:list, Sy:list, Sz:list)->[float, float, float]:
        Sx_expec = expectation(rho, num_qubits, Sx)
        Sy_expec = expectation(rho, num_qubits, Sy)
        Sz_expec = expectation(rho, num_qubits, Sz)
        return -(Sx_expec * np.cos(x0[1]) * np.sin(x0[0]) + Sy_expec * np.sin(x0[1]) * np.sin(x0[0]) + Sz_expec * np.cos(x0[0]))

    operators = construct_spin_mats(num_qubits)
    init_angles = np.array([np.pi, 2 * np.pi])
    max_polarization = 0
    for attempt in range(optimization_attempts):
        np.random.seed(int(time.time() * 10 ** 7) % 10 ** 7)
        out = mini(minimize_polarization, x0=init_angles * np.random.rand(2), args=(rho, num_qubits, *operators, ), method='Nelder-Mead', tol=1e-2, bounds=[(0, np.pi), (0, 2 * np.pi)])
        if out.fun < max_polarization:
            max_polarization = out.fun
            opt_angles = out.x

    # polarize rho along x
    out_state = state_integrator(rho, num_qubits, operators[-1], -opt_angles[1])

    cov_mat = np.zeros((2,2))
    for op1_idx, op1 in enumerate(operators[1:]):
        for op2_idx, op2 in enumerate(operators[1:]):
            cov_mat[op1_idx, op2_idx] = covariance(out_state, num_qubits, op1, op2)

    vals, vecs = eigen(cov_mat)
    return [num_qubits * np.min(vals)/collective_projection(out_state, num_qubits, operators), opt_angles[0], opt_angles[1]]

def collectivity(rho:list, num_qubits:int)->float:
    Jmax = int(np.ceil(num_qubits/2))
    dist = distribution(rho, num_qubits)

    running_sum = 0
    for Jth_sector in range(Jmax + 1):
        for mz_idx, mz in enumerate(range(-Jth_sector, Jth_sector + 1)):
            # sum of probabilities over J and mz of P(J, mz) * J * (J + 1)
            running_sum += dist[Jth_sector][mz] * Jth_sector * (Jth_sector + 1)

    return running_sum

def distribution(rho:list, num_qubits:int)->dict:
    '''
    construct a state's z-distribution across all Jth sectors
    :param rho:
    :param num_qubits:
    :return: dictionary of Jth sector's dictionary containing probabilities to be in the mzth projection
    '''
    Jmax = int(np.ceil(num_qubits / 2))
    Sz = construct_Jz(num_qubits)
    distribution_across_sectors = {}

    for Jth_sector in range(Jmax + 1):
        distribution_across_sectors[Jth_sector]={}
        for mz in range(2 * Jth_sector + 1):
            # over the Jth sector . . . . . . . . . . . grab mz . . . . . . . . . . . . . . . . and set P(J, mz)
            distribution_across_sectors[Jth_sector][Sz[Jth_sector][mz, mz].real] = rho[Jth_sector][mz,mz].real

    return distribution_across_sectors

def husimi(rho:list, num_qubits:int, resolution_pts:int=101)->None:
    def construct_coherent(num_qubits: int, theta: float = np.pi, phi: float = 0):
        '''
        Construct a coherent state with given theta, phi within a sector of the Dicke manifold.
        '''
        Jmax = int(np.ceil(num_qubits / 2))
        full_state = []
        for Jth_sector in range(Jmax + 1):
            coherentState = np.zeros((2 * Jth_sector + 1, 1), dtype='complex128')
            for k in range(2 * Jth_sector + 1):
                coherentState[k, 0] = (np.exp(1j * phi) * np.tan(theta / 2)) ** k * np.sqrt(binom(2 * Jth_sector, k))

            full_state.append(coherentState * np.cos(theta / 2) ** (2 * Jth_sector))

        return full_state

    def compute_projection(num_qubits: int, full_distribution: dict, theta: float, phi: float) -> np.ndarray:
        '''
        For a given theta and phi sum the total state projection weighted by the total probability of being in the
        Jth_sector.
        :param full_distribution: Distribution of 2J+1 mz projections across all J = {N/2, N/2 - 1, . . ., 1, 0} state sectors.
        '''

        Jmax = int(np.ceil(num_qubits / 2))
        projection = 0
        sector_probability_threshold = (1e-1) / 2
        for Jth_sector in range(Jmax + 1):
            sector_probability = sum(list(full_distribution[Jth_sector].values()))
            if sector_probability < sector_probability_threshold:
                continue

            coherent = construct_coherent(num_qubits, theta, phi)
            projection += (coherent[Jth_sector].T.conj() @ rho[Jth_sector] @ coherent[
                Jth_sector]).real * sector_probability

        return projection

    full_dist = distribution(rho, num_qubits)

    theta, phi = np.meshgrid(np.linspace(0, np.pi, resolution_pts), np.linspace(0, 2 * np.pi, resolution_pts))
    projection = np.vectorize(functools.partial(compute_projection, num_qubits, full_dist))(theta, phi)
    vmax = np.max(abs(projection))
    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=0)
    cmap = plt.get_cmap('inferno')(norm(projection))

    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)

    fig = plt.figure()
    ax = [fig.add_subplot(111, projection='3d')]

    for axis, side in zip(ax, [1, -1]):
        axis.plot_surface(side * x, side * y, z, rstride=1, cstride=1, facecolors=cmap, rasterized=True, shade=False)

    elev, azim = 0, 0
    for axis in ax:
        axis.set_xlim((-0.7, 0.7))
        axis.set_ylim((-0.7, 0.7))
        axis.set_zlim((-0.8, 0.8))
        axis.view_init(elev=elev, azim=azim)
        axis.set_axis_off()

    left = -0.01
    right = 1
    bottom = -0.03
    top = 1
    rect = (left, bottom, right, top)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0, rect=rect)
    return fig, ax
