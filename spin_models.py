#!/usr/bin/env python3
import argparse
import functools
import sys
from typing import Optional

import numpy as np
import scipy

from dissipation import Dissipator

# qubit/spin states
ket_0 = np.array([1, 0])  # |0>, spin up
ket_1 = np.array([0, 1])  # |1>, spin down

# Pauli operators
pauli_Z = np.array([[1, 0], [0, -1]])  # |0><0| - |1><1|
pauli_X = np.array([[0, 1], [1, 0]])  # |0><1| + |1><0|
pauli_Y = -1j * pauli_Z @ pauli_X


def simulate_sensing_protocol(
    num_qubits: int,
    entangling_hamiltonian: np.ndarray,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = "XYZ",
) -> np.ndarray:
    """
    Simulate a sensing protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about the X axis by the angle 'params[0] * np.pi' (with Hamiltonian 'Sx').
    2. Evolve under a given entangling Hamiltonian for time 'params[1] * np.pi * num_qubits'.
    3. Rotate about the axis 'X_phi' by the angle '-params[2] * np.pi',
       where 'phi = params[3] * 2 * np.pi' and 'X_phi = cos(phi) X + sin(phi) Y'.

    If dissipation_rates is nonzero, qubits experience dissipation during the squeezing step (2).
    See the documentation for the Dissipator class for a general explanation of the
    dissipation_rates and dissipation_format arguments.

    This method additionally divides the dissipator (equivalently, all dissipation rates) by a
    factor of 'np.pi * num_qubits' in order to "normalize" dissipation time scales, and make them
    comparable to the time scales of coherent evolution.  Dividing a Dissipator with dissipation
    rates 'r' by a factor of 'np.pi * num_qubits' makes so that each qubit depolarizes with
    probability 'e^(-params[1] * r)' by the end of the OAT protocol.
    """
    assert len(params) == 4, "must provide 4 simulation parameters!"

    # construct collective spin operators
    collective_Sx, collective_Sy, collective_Sz = collective_spin_ops(num_qubits)

    # rotate the all-|1> state about the X axis
    time_1 = params[0] * np.pi
    qubit_ket = np.sin(time_1 / 2) * ket_0 + 1j * np.cos(time_1 / 2) * ket_1
    qubit_state = np.outer(qubit_ket, qubit_ket.conj())
    state_1 = functools.reduce(np.kron, [qubit_state] * num_qubits)

    # entangle!
    time_2 = params[1] * np.pi * num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, entangling_hamiltonian, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[2] * np.pi
    rot_axis_angle = params[3] * 2 * np.pi
    final_hamiltonian = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_3, final_hamiltonian)

    return state_3


def simulate_OAT(
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = "XYZ",
) -> np.ndarray:
    """Simulate a one-axis twisting (OAT) protocol."""
    _, _, collective_Sz = collective_spin_ops(num_qubits)
    hamiltonian = collective_Sz.diagonal() ** 2 / num_qubits
    return simulate_sensing_protocol(num_qubits, hamiltonian, params, dissipation_rates, dissipation_format)


def simulate_TAT(
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = "XYZ",
) -> np.ndarray:
    """Simulate a two-axis twisting (TAT) protocol."""
    collective_Sx, collective_Sy, _ = collective_spin_ops(num_qubits)
    hamiltonian = (collective_Sx @ collective_Sy + collective_Sy @ collective_Sx) / num_qubits
    return simulate_sensing_protocol(num_qubits, hamiltonian, params, dissipation_rates, dissipation_format)


def evolve_state(
    density_op: np.ndarray,
    time: float,
    hamiltonian: np.ndarray,
    dissipator: Optional[Dissipator] = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> np.ndarray:
    """
    Time-evolve a given initial density operator for a given amount of time under the given Hamiltonian and (optionally) Dissipator.
    """
    if time == 0:
        return density_op
    if time < 0:
        time, hamiltonian = -time, -hamiltonian
    solution = scipy.integrate.solve_ivp(
        time_deriv,
        [0, time],
        density_op.ravel(),
        t_eval=[time],
        rtol=rtol,
        atol=atol,
        method="DOP853",
        args=(hamiltonian, dissipator),
    )
    final_vec = solution.y[:, -1]
    return final_vec.reshape(density_op.shape)


def time_deriv(
    _: float,
    density_op: np.ndarray,
    hamiltonian: np.ndarray,
    dissipator: Optional[Dissipator] = None,
) -> np.ndarray:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing Markovian evolution.

    The first argument is a time parameter, indicating this function should return the time derivative of 'density_op' at a particular time.
    The time parameter is not used here, but it is necessary for compatibility with scipy.integrate.solve_ivp.
    """
    # coherent evolution
    if hamiltonian.ndim == 2:
        # ... computed with ordinary matrix multiplication
        density_op.shape = hamiltonian.shape
        output = -1j * (hamiltonian @ density_op - density_op @ hamiltonian)
    else:
        # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
        # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
        density_op.shape = hamiltonian.shape * 2
        output = -1j * (hamiltonian[:, np.newaxis] * density_op - density_op * hamiltonian)

    # dissipation
    if dissipator:
        output += dissipator @ density_op

    density_op.shape = (-1,)
    return output.ravel()


@functools.cache
def collective_spin_ops(num_qubits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct collective spin operators."""
    return (
        collective_op(pauli_X, num_qubits) / 2,
        collective_op(pauli_Y, num_qubits) / 2,
        collective_op(pauli_Z, num_qubits) / 2,
    )


def collective_op(op: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum((op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits)), start=np.array(0))


def op_on_qubit(op: np.ndarray, qubit: int, total_qubit_num: int) -> np.ndarray:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the identity operator) on all other qubits.
    """
    iden_before = np.eye(2**qubit, dtype=op.dtype)
    iden_after = np.eye(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    return np.kron(np.kron(iden_before, op), iden_after)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--dissipation", type=float, default=0.0)
    parser.add_argument("--params", type=float, nargs=4, default=[0.5, 0.5, 0.5, 0])
    args = parser.parse_args(sys.argv[1:])

    # simulate the OAT potocol
    final_state = simulate_OAT(args.num_qubits, args.params, args.dissipation)

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
