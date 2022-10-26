#!/usr/bin/env python3
import argparse
import functools
import sys

import numpy as np
import scipy

# Pauli operators
pauli_Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]])
pauli_X = scipy.sparse.csr_matrix([[0, 1], [1, 0]])
pauli_Y = -1j * pauli_Z @ pauli_X


class LindbladianMap:
    """
    Data structure to store a Lindbladian, which is the time derivative operator for a density
    matrix undergoing Markovian time evolution.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray | scipy.sparse.spmatrix,
        *noise_data: tuple[float, np.ndarray | scipy.sparse.spmatrix],
    ) -> None:
        self._hamiltonian = hamiltonian
        self._jump_ops = [np.sqrt(noise_rate) * op for noise_rate, op in noise_data if noise_rate]
        self._recycling_ops = [op.conj().T @ op for op in self._jump_ops]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._hamiltonian.shape

    def __call__(self, density_op: np.ndarray) -> np.ndarray:
        """Return the action of this Lindbladian on the given density operator."""
        coherent_part = -1j * (self._hamiltonian @ density_op - density_op @ self._hamiltonian)
        if self._jump_ops:
            jump_part = sum(op @ density_op @ op.conj().T for op in self._jump_ops)
            recycling_part = sum(op @ density_op + density_op @ op for op in self._recycling_ops)
            return coherent_part + jump_part - recycling_part / 2
        return coherent_part


def op_on_qubit(op: scipy.sparse.spmatrix, qubit: int, total_qubit_num: int) -> scipy.sparse.spmatrix:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the
    identity operator) on all other qubits.
    """
    ops = [scipy.sparse.eye(2, dtype=int)] * total_qubit_num
    ops[qubit] = op
    net_op = functools.reduce(scipy.sparse.kron, ops).tocsr()
    net_op.eliminate_zeros()
    return net_op


def collective_op(op: scipy.sparse.spmatrix, num_qubits: int) -> scipy.sparse.spmatrix:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits))


def time_deriv(_: float, density_op: np.ndarray, lindbladian_map: LindbladianMap) -> np.ndarray:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
    Markovian evolution with the given Lindbladian.

    The first argument is blank to integrate with scipy.integrate.solve_ivp.
    """
    inp_matrix = density_op.reshape(lindbladian_map.input_shape)
    out_matrix = lindbladian_map(inp_matrix)
    return out_matrix.ravel()


def evolve_state(
    density_op: np.ndarray,
    time: float,
    hamiltonian: np.ndarray | scipy.sparse.spmatrix,
    *noise_data: tuple[float, np.ndarray | scipy.sparse.spmatrix],
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
        args=(LindbladianMap(hamiltonian, *noise_data),),
    )
    final_vec = scipy.integrate.solve_ivp(*args, **kwargs).y[:, -1]
    return final_vec.reshape(density_op.shape)


def simulate_OAT(
    num_qubits: int, params: tuple[float, float, float, float] | np.ndarray, noise_level: float = 0
) -> np.ndarray:
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

    # collective spin operators
    collective_Sx = collective_op(pauli_X, num_qubits) / 2
    collective_Sy = collective_op(pauli_Y, num_qubits) / 2
    collective_Sz = collective_op(pauli_Z, num_qubits) / 2

    if noise_level:
        # collect noise data as a list of tuples: (jump_rate, jump_operator)
        depolarizing_rate = noise_level / (np.pi * num_qubits)
        noise_data = [
            (depolarizing_rate, op_on_qubit(pauli / 2, qubit, num_qubits))
            for pauli in [pauli_X, pauli_Y, pauli_Z]
            for qubit in range(num_qubits)
        ]
    else:
        noise_data = []

    # initialize a state pointing down along Z (all qubits in |0>)
    state_0 = np.zeros((2**num_qubits,) * 2, dtype=complex)
    state_0[-1, -1] = 1

    # rotate about the X axis
    time_0 = params[0] * np.pi
    hamiltonian_0 = collective_Sx
    state_1 = evolve_state(state_0, time_0, hamiltonian_0, *noise_data)

    # squeeze!
    time_1 = params[1] * np.pi * num_qubits
    hamiltonian_1 = collective_Sz @ collective_Sz / num_qubits
    state_2 = evolve_state(state_1, time_1, hamiltonian_1, *noise_data)

    # un-rotate about a chosen axis
    time_2 = -params[2] * np.pi
    rot_axis_angle = params[3] * np.pi / 2
    hamiltonian_2 = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_2, hamiltonian_2, *noise_data)

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
    final_pauli_vars = [
        (final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)
    ]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
