#!/usr/bin/env python3
import argparse
import string
import sys
from typing import Iterable

import numpy as np
import scipy

# Pauli operators
pauli_Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]])
pauli_X = scipy.sparse.csr_matrix([[0, 1], [1, 0]])
pauli_Y = -1j * pauli_Z @ pauli_X


def tensordot_insert(axes: int | Iterable[int], tensor_A: np.ndarray, tensor_B: np.ndarray) -> np.ndarray:
    """
    Contract and "insert" tensor_A into tensor_B at the given axes,
    e.g. A_{J,L,j,l} B_{i,j,k,l,m} -> C_{i,J,k,L,m} (with axes=[1,3])
    """
    if isinstance(axes, int):
        axes = [axes]

    indices_B = string.ascii_lowercase[: tensor_B.ndim]
    old_axis_indices = [indices_B[axis] for axis in axes]
    new_axis_indices = [index.upper() for index in old_axis_indices]
    indices_A = "".join(new_axis_indices + old_axis_indices)

    indices_C = indices_B
    for old_idx, new_idx in zip(old_axis_indices, new_axis_indices):
        indices_C = indices_C.replace(old_idx, new_idx)
    contraction = f"{indices_A},{indices_B}->{indices_C}"
    return np.einsum(contraction, tensor_A, tensor_B)


class Dissipator:
    """Data structure for characterizing noise processes."""

    def __init__(self, num_qubits: int, noise_level: float | tuple[float, float, float]) -> None:
        self._input_shape = (2,) * (num_qubits * 2)
        self._num_qubits = num_qubits
        if isinstance(noise_level, tuple):
            noise_x, noise_y, noise_z = noise_level
        else:
            noise_x = noise_y = noise_z = noise_level
        self._qubit_jump_ops = [np.sqrt(noise_x) * pauli_X.todense() / 2, np.sqrt(noise_y) * pauli_Y.todense() / 2, np.sqrt(noise_z) * pauli_Z.todense() / 2]
        self._qubit_recycling_ops = [op.conj().T @ op for op in self._qubit_jump_ops]
        self._is_trivial = noise_x == noise_y == noise_z == 0

    @property
    def is_trivial(self) -> bool:
        return self._is_trivial

    def __matmul__(self, density_op: np.ndarray) -> np.ndarray:
        input_shape = density_op.shape
        density_tensor = density_op.reshape(self._input_shape)

        output = np.zeros_like(density_tensor)
        for jump_op, recycling_op in zip(self._qubit_jump_ops, self._qubit_recycling_ops):
            for qubit in range(self._num_qubits):
                output += tensordot_insert(qubit, jump_op, tensordot_insert(self._num_qubits + qubit, jump_op.conj().T, density_tensor))
                output -= sum(tensordot_insert(axis, recycling_op, density_tensor) for axis in [qubit, self._num_qubits + qubit]) / 2

        return output.reshape(input_shape)


class Lindbladian:
    """Data structure to store a Lindbladian operator, which is the time derivative operator for a density matrix undergoing Markovian time evolution."""

    def __init__(
        self,
        hamiltonian: np.ndarray | scipy.sparse.spmatrix,
        noise_level: float | tuple[float, float, float],
    ) -> None:
        self._input_shape = hamiltonian.shape
        self._hamiltonian = hamiltonian

        num_qubits = int(np.log2(self._hamiltonian.shape[0]))
        self._dissipator = Dissipator(num_qubits, noise_level)

    def __matmul__(self, density_op: np.ndarray) -> np.ndarray:
        """Return the action of this Lindbladian on the given density operator."""
        input_shape = density_op.shape
        density_op = density_op.reshape(self._input_shape)

        output = -1j * (self._hamiltonian @ density_op - density_op @ self._hamiltonian)
        if not self._dissipator.is_trivial:
            output += self._dissipator @ density_op

        return output.reshape(input_shape)


def op_on_qubit(op: scipy.sparse.spmatrix, qubit: int, total_qubit_num: int) -> scipy.sparse.spmatrix:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the
    identity operator) on all other qubits.
    """
    iden_before = scipy.sparse.identity(2**qubit, dtype=op.dtype)
    iden_after = scipy.sparse.identity(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    return scipy.sparse.kron(scipy.sparse.kron(iden_before, op), iden_after)


def collective_op(op: scipy.sparse.spmatrix, num_qubits: int) -> scipy.sparse.spmatrix:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits))


def time_deriv(_: float, density_op: np.ndarray, lindbladian: Lindbladian) -> np.ndarray:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
    Markovian evolution with the given Lindbladian.

    The first argument is blank to integrate with scipy.integrate.solve_ivp.
    """
    return lindbladian @ density_op


def evolve_state(
    density_op: np.ndarray,
    time: float,
    hamiltonian: np.ndarray | scipy.sparse.spmatrix,
    noise_level: float | tuple[float, float, float],
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
        args=(Lindbladian(hamiltonian, noise_level),),
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
    assert noise_level >= 0, "noise levels cannot be negative!"
    assert len(params) == 4, "must provide 4 parameters!"

    # normalize noise level
    noise_level /= np.pi * num_qubits

    # collective spin operators
    collective_Sx = collective_op(pauli_X, num_qubits) / 2
    collective_Sy = collective_op(pauli_Y, num_qubits) / 2
    collective_Sz = collective_op(pauli_Z, num_qubits) / 2

    # initialize a state pointing down along Z (all qubits in |0>)
    state_0 = np.zeros((2**num_qubits,) * 2, dtype=complex)
    state_0[-1, -1] = 1

    # rotate about the X axis
    time_0 = params[0] * np.pi
    hamiltonian_0 = collective_Sx
    state_1 = evolve_state(state_0, time_0, hamiltonian_0, noise_level)

    # squeeze!
    time_1 = params[1] * np.pi * num_qubits
    hamiltonian_1 = collective_Sz @ collective_Sz / num_qubits
    state_2 = evolve_state(state_1, time_1, hamiltonian_1, noise_level)

    # un-rotate about a chosen axis
    time_2 = -params[2] * np.pi
    rot_axis_angle = params[3] * np.pi / 2
    hamiltonian_2 = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_2, hamiltonian_2, noise_level)

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
