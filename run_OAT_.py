#!/usr/bin/env python3
import argparse
import functools
import sys
from typing import Callable, Optional

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

from dissipation_ import Dissipator

COMPLEX_DTYPE = tf.dtypes.complex128

# qubit/spin states
ket_0 = tf.constant([1, 0], shape=(2,), dtype=COMPLEX_DTYPE)  # |0>, spin up
ket_1 = tf.constant([0, 1], shape=(2,), dtype=COMPLEX_DTYPE)  # |1>, spin down

# Pauli operators
pauli_Z = tf.constant([[1, 0], [0, -1]], shape=(2, 2), dtype=COMPLEX_DTYPE)  # |0><0| - |1><1|
pauli_X = tf.constant([[0, 1], [1, 0]], shape=(2, 2), dtype=COMPLEX_DTYPE)  # |0><1| + |1><0|
pauli_Y = -1j * pauli_Z @ pauli_X


def simulate_OAT(
    num_qubits: int,
    params: tuple[float, float, float, float] | tf.Tensor,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = "XYZ",
) -> tf.Tensor:
    """
    Simulate a one-axis twisting (OAT) protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about the X axis by the angle 'params[0] * np.pi' (with Hamiltonian 'Sx').
    2. Squeeze with Hamiltonian 'Sz^2 / num_qubits' for time 'params[1] * np.pi * num_qubits'.
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
    qubit_state = tf.experimental.numpy.outer(qubit_ket, tf.math.conj(qubit_ket))
    state_1 = functools.reduce(tf.experimental.numpy.kron, [qubit_state] * num_qubits)

    # squeeze!
    time_2 = params[1] * np.pi * num_qubits
    hamiltonian_2 = tf.experimental.numpy.diagonal(collective_Sz) ** 2 / num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, hamiltonian_2, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[2] * np.pi
    rot_axis_angle = params[3] * 2 * np.pi
    hamiltonian_3 = np.cos(rot_axis_angle) * collective_Sx + np.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_3, hamiltonian_3)

    return state_3


def evolve_state(
    density_op: tf.Tensor,
    time: float,
    hamiltonian: tf.Tensor,
    dissipator: Optional[Dissipator] = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tf.Tensor:
    """
    Time-evolve a given initial density operator for a given amount of time under the given Hamiltonian and (optionally) Dissipator.
    """
    if time == 0:
        return density_op
    if time < 0:
        time, hamiltonian = -time, -hamiltonian
    time_deriv, time_deriv_jacobian = get_time_deriv_funs(hamiltonian, dissipator)

    from time import time as current_time

    start = current_time()

    # def _time_deriv(_, density_op):
    #     shape = (hamiltonian.shape[0],) * 2
    #     return time_deriv(_, density_op.reshape(shape)).numpy().ravel()

    # solution = scipy.integrate.solve_ivp(
    #     _time_deriv,
    #     [0, time],
    #     density_op.numpy().ravel(),
    #     t_eval=[time],
    #     rtol=rtol,
    #     atol=atol,
    #     method="DOP853",
    # )
    # print(current_time() - start)
    # final_vec = solution.y[:, -1]
    # return tf.constant(final_vec.reshape(density_op.shape))

    result = tfp.math.ode.DormandPrince(rtol=rtol, atol=atol).solve(
        time_deriv,
        0,
        density_op,
        solution_times=[0, time],
        jacobian_fn=time_deriv_jacobian,
    )
    print(current_time() - start)
    return result.states[-1]


def get_time_deriv_funs(
    hamiltonian: tf.Tensor, dissipator: Optional[Dissipator] = None
) -> tuple[Callable[[float, tf.Tensor], tf.Tensor], Callable[[float, tf.Tensor], tf.Tensor]]:
    """
    For a given Hamiltonian and dissipator, construct a method that returns the time derivative of a given density operator at a given time.
    Return this method, as well as its jacobian.
    """

    def time_deriv(_: float, density_op: tf.Tensor) -> tf.Tensor:
        # compute commutator with hamiltonian
        if hamiltonian.ndim == 2:
            # ... computed with ordinary matrix multiplication
            ham_bracket = hamiltonian @ density_op - density_op @ hamiltonian
        else:
            # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
            # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
            ham_bracket = hamiltonian[:, tf.newaxis] * density_op - density_op * hamiltonian
        if not dissipator:
            return -1j * ham_bracket
        return -1j * ham_bracket + dissipator @ density_op

    def time_deriv_jacobian(time: float, density_op: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(density_op)
            out_matrix = time_deriv(time, density_op)
        return tape.jacobian(out_matrix, density_op)

    return time_deriv, time_deriv_jacobian


@functools.cache
def collective_spin_ops(num_qubits: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Construct collective spin operators."""
    return (
        collective_op(pauli_X, num_qubits) / 2,
        collective_op(pauli_Y, num_qubits) / 2,
        collective_op(pauli_Z, num_qubits) / 2,
    )


def collective_op(op: tf.Tensor, num_qubits: int) -> tf.Tensor:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(
        (op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits)),
        start=tf.constant(0, dtype=op.dtype),
    )


def op_on_qubit(op: tf.Tensor, qubit: int, total_qubit_num: int) -> tf.Tensor:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the identity operator) on all other qubits.
    """
    iden_before = tf.eye(2**qubit, dtype=op.dtype)
    iden_after = tf.eye(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    return functools.reduce(tf.experimental.numpy.kron, [iden_before, op, iden_after])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--dissipation", type=float, default=0.0)
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
    final_pauli_vals = [tf.math.real(tf.linalg.trace(final_state @ op)).numpy() for op in mean_ops]
    final_pauli_vars = [
        tf.math.real(tf.linalg.trace(final_state @ (op @ op))).numpy() - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)
    ]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
