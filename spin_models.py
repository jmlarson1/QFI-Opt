#!/usr/bin/env python3
import argparse
import functools
import itertools
import sys
from typing import Optional

import numpy as np
import scipy

from dissipation import Dissipator

DEFAULT_DISSIPATION_FORMAT = "XYZ"

# qubit/spin states
KET_0 = np.array([1, 0])  # |0>, spin up
KET_1 = np.array([0, 1])  # |1>, spin down

# Pauli operators
PAULI_Z = np.array([[1, 0], [0, -1]])  # |0><0| - |1><1|
PAULI_X = np.array([[0, 1], [1, 0]])  # |0><1| + |1><0|
PAULI_Y = -1j * PAULI_Z @ PAULI_X


def simulate_sensing_protocol(
    num_qubits: int,
    entangling_hamiltonian: np.ndarray,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
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
    qubit_ket = np.sin(time_1 / 2) * KET_0 + 1j * np.cos(time_1 / 2) * KET_1
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
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a one-axis twisting (OAT) protocol."""
    _, _, collective_Sz = collective_spin_ops(num_qubits)
    hamiltonian = collective_Sz.diagonal() ** 2 / num_qubits
    return simulate_sensing_protocol(num_qubits, hamiltonian, params, dissipation_rates, dissipation_format)


def simulate_TAT(
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a two-axis twisting (TAT) protocol."""
    collective_Sx, collective_Sy, _ = collective_spin_ops(num_qubits)
    hamiltonian = (collective_Sx @ collective_Sy + collective_Sy @ collective_Sx) / num_qubits
    return simulate_sensing_protocol(num_qubits, hamiltonian, params, dissipation_rates, dissipation_format)


def simulate_spin_chain(
    coupling_op: np.ndarray,
    coupling_exponent: float,
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate an entangling protocol for a spin chain with power-law interactions."""
    normalization_factor = num_qubits * np.mean((1 / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)))
    hamiltonian = sum(
        act_on_subsystem(num_qubits, coupling_op, pp, qq) / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)
    )
    return simulate_sensing_protocol(num_qubits, hamiltonian / normalization_factor, params, dissipation_rates, dissipation_format)


def simulate_ising_chain(
    coupling_exponent: float,
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = np.kron(PAULI_Z, PAULI_Z) / 2
    return simulate_spin_chain(coupling_op, coupling_exponent, num_qubits, params, dissipation_rates, dissipation_format)


def simulate_XX_chain(
    coupling_exponent: float,
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)) / 2
    return simulate_spin_chain(coupling_op, coupling_exponent, num_qubits, params, dissipation_rates, dissipation_format)


def simulate_local_TAT_chain(
    coupling_exponent: float,
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_Y) + np.kron(PAULI_Y, PAULI_X)) / 2
    return simulate_spin_chain(coupling_op, coupling_exponent, num_qubits, params, dissipation_rates, dissipation_format)


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
        collective_op(PAULI_X, num_qubits) / 2,
        collective_op(PAULI_Y, num_qubits) / 2,
        collective_op(PAULI_Z, num_qubits) / 2,
    )


def collective_op(op: np.ndarray, num_qubits: int) -> np.ndarray:
    """Compute the collective version of a single-qubit qubit operator: sum_q op_q."""
    assert op.shape == (2, 2)
    return sum((act_on_subsystem(num_qubits, op, qubit) for qubit in range(num_qubits)), start=np.array(0))


def act_on_subsystem(num_qubits: int, op: np.ndarray, *qubits: int) -> np.ndarray:
    """
    Return an operator that acts with 'op' in the given qubits, and trivially (with the identity operator) on all other qubits.
    """
    assert op.shape == (2 ** len(qubits),) * 2, "Operator shape {op.shape} is inconsistent with the number of target qubits provided, {num_qubits}!"
    identity = np.eye(2 ** (num_qubits - len(qubits)), dtype=op.dtype)
    system_op = np.kron(op, identity)

    # rearrange operator into tensor factors addressing each qubit
    system_op = np.moveaxis(
        system_op.reshape((2,) * 2 * num_qubits),
        range(num_qubits),
        range(0, 2 * num_qubits, 2),
    ).reshape((4,) * num_qubits)

    # move the first len(qubits) tensor factors to the target qubits
    system_op = np.moveaxis(
        system_op,
        range(len(qubits)),
        qubits,
    )

    # split and re-combine tensor factors again to recover the operator as a matrix
    return np.moveaxis(
        system_op.reshape((2,) * 2 * num_qubits),
        range(0, 2 * num_qubits, 2),
        range(num_qubits),
    ).reshape((2**num_qubits,) * 2)


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
    mean_X = collective_op(PAULI_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(PAULI_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(PAULI_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_PAULI_vals = [(final_state @ op).trace().real for op in mean_ops]
    final_PAULI_vars = [(final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_PAULI_vals)]
    print("[<X>, <Y>, <Z>]:", final_PAULI_vals)
    print("[var(X), var(Y), var(Z)]:", final_PAULI_vars)
