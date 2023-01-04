#!/usr/bin/env python3
import argparse
import functools
import sys
from typing import Optional

import numpy as np
import scipy

from dissipation import Dissipator

# qubit states
ket_0 = np.array([0, 1])
ket_1 = np.array([1, 0])

# Pauli operators
pauli_Z = scipy.sparse.dia_matrix([[1, 0], [0, -1]])  # |1><1| - |0><0|
pauli_X = scipy.sparse.csr_matrix([[0, 1], [1, 0]])  # |0><1| + |1><0|
pauli_Y = -1j * pauli_Z @ pauli_X


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


@functools.cache
def collective_spin_ops(num_qubits: int) -> tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix, scipy.sparse.spmatrix]:
    """Construct collective spin operators."""
    collective_Sx = collective_op(pauli_X, num_qubits) / 2
    collective_Sy = collective_op(pauli_Y, num_qubits) / 2
    collective_Sz = collective_op(pauli_Z, num_qubits) / 2
    return collective_Sx, collective_Sy, collective_Sz


def time_deriv(_: float, density_op: np.ndarray, hamiltonian: np.ndarray | scipy.sparse.spmatrix, dissipator: Optional[Dissipator] = None) -> np.ndarray:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing Markovian evolution.
    The first argument is blank to integrate with scipy.integrate.solve_ivp.
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
        output = -1j * ((hamiltonian * density_op.T).T - density_op * hamiltonian)

    # dissipation
    if dissipator:
        output += dissipator @ density_op

    density_op.shape = (-1,)
    return output.ravel()


def evolve_state(
    density_op: np.ndarray,
    time: float,
    hamiltonian: np.ndarray | scipy.sparse.spmatrix,
    dissipator: Optional[Dissipator] = None,
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


def simulate_OAT(
    num_qubits: int,
    params: tuple[float, float, float, float] | np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = "XYZ",
) -> np.ndarray:
    """
    Simulate a one-axis twisting (OAT) protocol, and return the final state (density matrix).

    Starting with an initial all-|0> state (all spins pointing down along the Z axis):
    1. Rotate about the X axis by the angle 'params[0] * np.pi' (with Hamiltonian 'Sx').
    2. Squeeze with Hamiltonian 'Sz^2 / num_qubits' for time 'params[1] * np.pi * num_qubits'.
    3. Rotate about the axis 'X_phi' by the angle '-params[2] * np.pi',
       where 'phi = params[3] * np.pi / 2' and 'X_phi = cos(phi) X + sin(phi) Y'.

    If dissipation_rates is nonzero, qubits experience dissipation during the squeezing step (2).
    See the documentation for the Dissipator class for a general explanation of the
    dissipation_rates and dissipation_format arguments, but note that this method additionally
    divides the dissipator (equivalently, the dissipation rates) by a factor of 'np.pi * num_qubits'
    in order to "normalize" dissipation time scales and make them comparable to the time scales of
    coherent evolution.  There are two ways to interpret this normalization of the dissipation
    rates:
    - Dividing the "bare" dissipation rate 'r' by a factor of 'np.pi' makes it so that XYZ-type
      dissipation (with rates 'r / np.pi') depolarizes a single qubit with probability 'e^(-r)' in
      time 'np.pi', or equivalently the time it takes for the Hamiltonian 'Sx' to flip a qubit.
      The additional divisor of 'num_qubits' accounts for the fact that the OAT protocol takes time
      'O(num_qubits)' when 'params[1] ~ O(1)', so without this divisor dynamics would be completely
      dominated by dissipation when 'num_qubits >> 1'.
    - Dividing the "bare" dissipation rate 'r' by a factor of 'np.pi * num_qubits' makes so that
      each qubit depolarizes with probability 'e^(-params[1])' by the end of the OAT protocol.
    """
    assert len(params) == 4, "must provide 4 simulation parameters!"

    # construct collective spin operators
    collective_Sx, collective_Sy, collective_Sz = collective_spin_ops(num_qubits)

    # rotate the all-|0> state about the X axis
    time_1 = params[0] * np.pi
    qubit_ket = np.cos(time_1 / 2) * ket_0 - 1j * np.sin(time_1 / 2) * ket_1
    qubit_state = np.outer(qubit_ket, qubit_ket.conj())
    state_1 = functools.reduce(np.kron, [qubit_state] * num_qubits)

    # squeeze!
    time_2 = params[1] * np.pi * num_qubits
    hamiltonian_2 = collective_Sz.diagonal() ** 2 / num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, hamiltonian_2, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[2] * np.pi
    rot_axis_angle = params[3] * np.pi / 2
    hamiltonian_3 = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_3, hamiltonian_3)

    return state_3


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--dissipation", type=float, default=0)
    parser.add_argument("--params", type=float, nargs=4, required=True)
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
