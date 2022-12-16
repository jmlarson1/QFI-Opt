#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import scipy

# Pauli operators
pauli_Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]])
pauli_X = scipy.sparse.csr_matrix([[0, 1], [1, 0]])
pauli_Y = -1j * pauli_Z @ pauli_X


def log2_int(val: int) -> int:
    return len(bin(val)) - 2


def conjugate_by_X(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'X_q rho X_q'."""
    ndim = log2_int(density_op.size)
    tensor_shape = (2,) * ndim
    return np.flip(np.flip(density_op.reshape(tensor_shape), qubit), ndim // 2 + qubit).reshape(density_op.shape)


def conjugate_by_Z(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'Z_q rho Z_q'."""
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2 ** qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 0, :, :, 0, :]
    output[:, 0, :, :, 1, :] = -density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 0, :] = -density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 1, :] = density_op[:, 1, :, :, 1, :]
    density_op.shape = input_shape
    return output.reshape(input_shape)


def conjugate_by_Y(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'Y_q rho Y_q'."""
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2 ** qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :]
    output[:, 0, :, :, 1, :] = -density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 0, :] = -density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 1, :] = density_op[:, 0, :, :, 0, :]
    density_op.shape = input_shape
    return output.reshape(input_shape)


class Dissipator:
    """
    Data structure for representing a dissipation operator.
    Currently only allows for single-qubit depolarizing noise.
    """

    def __init__(self, depolarizing_rate: float | tuple[float, float, float]) -> None:
        if isinstance(depolarizing_rate, tuple):
            self._rate_x, self._rate_y, self._rate_z = tuple(rate / 4 for rate in depolarizing_rate)
        else:
            self._rate_x = self._rate_y = self._rate_z = depolarizing_rate / 4
        self._rate_sum = self._rate_x + self._rate_y + self._rate_z
        self._is_trivial = self._rate_x == self._rate_y == self._rate_z == 0

    @property
    def is_trivial(self) -> bool:
        return self._is_trivial

    def __matmul__(self, density_op: np.ndarray) -> np.ndarray:
        num_qubits = log2_int(density_op.size) // 2
        term_x = self._rate_x * sum(conjugate_by_X(density_op, qubit) for qubit in range(num_qubits))
        term_y = self._rate_y * sum(conjugate_by_Y(density_op, qubit) for qubit in range(num_qubits))
        term_z = self._rate_z * sum(conjugate_by_Z(density_op, qubit) for qubit in range(num_qubits))
        return term_x + term_y + term_z - self._rate_sum * num_qubits * density_op


def op_on_qubit(op: scipy.sparse.spmatrix, qubit: int, total_qubit_num: int) -> scipy.sparse.spmatrix:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the identity operator) on all other qubits.
    """
    iden_before = scipy.sparse.identity(2**qubit, dtype=op.dtype)
    iden_after = scipy.sparse.identity(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    return scipy.sparse.kron(scipy.sparse.kron(iden_before, op), iden_after)


def collective_op(op: scipy.sparse.spmatrix, num_qubits: int) -> scipy.sparse.spmatrix:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits))


def time_deriv(_: float, density_op: np.ndarray, hamiltonian: np.ndarray | scipy.sparse.spmatrix, dissipator: Dissipator) -> np.ndarray:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing Markovian evolution.
    The first argument is blank to integrate with scipy.integrate.solve_ivp.
    """
    density_op.shape = hamiltonian.shape
    output = -1j * (hamiltonian @ density_op - density_op @ hamiltonian)
    if not dissipator.is_trivial:
        output += dissipator @ density_op
    density_op.shape = (-1,)
    return output.ravel()


def evolve_state(
    density_op: np.ndarray,
    time: float,
    hamiltonian: np.ndarray | scipy.sparse.spmatrix,
    dissipator: Dissipator,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> np.ndarray:
    """Evolve a density operator under a Lindbladian for a given amount of time."""
    if time == 0:
        return density_op
    if time < 0:
        time, hamiltonian = -time, -hamiltonian
    args = (time_deriv, [0, time], density_op.ravel())
    kwargs = dict(
        t_eval=[time],
        rtol=rtol,
        atol=atol,
        method="DOP853",
        args=(hamiltonian, dissipator),
    )
    final_vec = scipy.integrate.solve_ivp(*args, **kwargs).y[:, -1]
    return final_vec.reshape(density_op.shape)


def simulate_OAT(num_qubits: int, params: tuple[float, float, float, float] | np.ndarray, noise_level: float = 0) -> np.ndarray:
    """
    Simulate a one-axis twisting (OAT) protocol, and return the final state (density matrix).

    Starting with an initial all-|0> state (all spins pointing down along the Z axis):
    1. Rotate about the X axis by the angle 'params[0] * np.pi' (with Hamiltonian Sx).
    2. Squeeze with Hamiltonian 'Sz^2 / num_qubits' for time 'params[1] * np.pi * num_qubits'.
    3. Rotate about the axis X_phi by the angle '-params[2] * np.pi',
       where phi = 'params[3] * np.pi / 2' and X_phi = cos(phi) X + sin(phi) Y.

    If noise_level > 0, qubits depolarize at a constant rate throughout the protocol.
    The depolarizing rate is chosen such that a single qubit (with num_qubits = 1) would depolarize
    with probability e^(-noise_level) in time pi (i.e., the time it takes to flip a spin with the
    Hamiltonian Sx).  The depolarizing rate is additionally reduced by a factor of num_qubits
    because the OAT protocol takes time O(num_qubits) when params[1] ~ O(1).
    """
    assert noise_level >= 0, "noise_level cannot be negative!"
    assert len(params) == 4, "must provide 4 parameters!"

    # construct collective spin operators
    collective_Sx = collective_op(pauli_X, num_qubits) / 2
    collective_Sy = collective_op(pauli_Y, num_qubits) / 2
    collective_Sz = collective_op(pauli_Z, num_qubits) / 2

    # construct the dissipator
    depolarizing_rate = noise_level / (np.pi * num_qubits)
    dissipator = Dissipator(depolarizing_rate)

    # initialize a state pointing down along Z (all qubits in |0>)
    state_0 = np.zeros((2**num_qubits,) * 2, dtype=complex)
    state_0[-1, -1] = 1

    # rotate about the X axis
    time_0 = params[0] * np.pi
    hamiltonian_0 = collective_Sx
    state_1 = evolve_state(state_0, time_0, hamiltonian_0, dissipator)

    # squeeze!
    time_1 = params[1] * np.pi * num_qubits
    hamiltonian_1 = collective_Sz @ collective_Sz / num_qubits
    state_2 = evolve_state(state_1, time_1, hamiltonian_1, dissipator)

    # un-rotate about a chosen axis
    time_2 = -params[2] * np.pi
    rot_axis_angle = params[3] * np.pi / 2
    hamiltonian_2 = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_2, hamiltonian_2, dissipator)

    return state_3


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0)
    parser.add_argument("--params", type=float, nargs=4, required=True)
    args = parser.parse_args(sys.argv[1:])

    # simulate the OAT potocol
    final_state = simulate_OAT(args.num_qubits, args.params, args.noise_level)

    # compute collective Pauli operators
    mean_X = collective_op(pauli_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(pauli_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(pauli_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_pauli_vals = [(final_state @ op).trace().real for op in mean_ops]
    final_pauli_vars = [(final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
