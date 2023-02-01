#!/usr/bin/env python3
import argparse
import sys

import tensorflow as tf
import tensorflow_probability as tfp
import numpy

# Pauli operators
pauli_Z = tf.constant([[1, 0], [0, -1]], shape=(2, 2), dtype=tf.dtypes.complex128)
pauli_X = tf.constant([[0, 1], [1, 0]], shape=(2, 2), dtype=tf.dtypes.complex128)
pauli_Y = -1j * tf.matmul(pauli_Z, pauli_X)


class LindbladianMap:
    """
    Data structure to store a Lindbladian, which is the time derivative operator for a density
    matrix undergoing Markovian time evolution.
    """

    def __init__(
        self,
        hamiltonian: tf.Tensor, 
        *dissipation_data: tf.Tensor,
    ) -> None:
        self._hamiltonian = hamiltonian
        self._jump_ops = [tf.complex(tf.math.sqrt(rate), tf.constant(0.0, dtype=tf.dtypes.float64)) * op for rate, op in dissipation_data if rate]
        self._recycling_ops = [tf.linalg.matmul(tf.transpose(tf.math.conj(op)), op) for op in self._jump_ops]

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._hamiltonian.shape

    def __call__(self, density_op: tf.Tensor) -> tf.Tensor:
        """Return the action of this Lindbladian on the given density operator."""
        coherent_part = -1j * (tf.linalg.matmul(self._hamiltonian, density_op) - tf.linalg.matmul(density_op, self._hamiltonian))
        if self._jump_ops:
            jump_part = sum(op @ density_op @ tf.transpose(tf.math.conj(op)) for op in self._jump_ops)
            recycling_part = sum(op @ density_op + density_op @ op for op in self._recycling_ops)
            return coherent_part + jump_part - recycling_part / 2
        return coherent_part


def op_on_qubit(op: tf.sparse.SparseTensor, qubit: int, total_qubit_num: int) -> tf.sparse.SparseTensor:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the
    identity operator) on all other qubits.
    """
    iden_before = tf.eye(2**qubit, dtype=op.dtype)
    iden_after = tf.eye(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    inter = tf.experimental.numpy.kron(iden_before, op)
    return tf.experimental.numpy.kron(inter, iden_after)


def collective_op(op: tf.Tensor, num_qubits: int) -> tf.Tensor:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits))


def evolve_state(
    density_op: tf.Tensor,
    time: float,
    hamiltonian: tf.Tensor | tf.sparse.SparseTensor,
    *dissipation_data: tuple[float, tf.Tensor  | tf.sparse.SparseTensor],
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tf.Tensor:
    """Evolve a density operator under a Lindbladian for a given amount of time."""
    if time == 0:
        return density_op
    if time < 0:
        time, hamiltonian = -time, -hamiltonian
    lindbladian_map = LindbladianMap(hamiltonian, *dissipation_data)

    def time_deriv(_: float, density_op: tf.Tensor,) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """
        inp_matrix = tf.reshape(density_op, lindbladian_map.input_shape)
        out_matrix = lindbladian_map(inp_matrix)
        return tf.experimental.numpy.ravel(out_matrix)

    def re_time_deriv(_: float, density_op: tf.Tensor) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """
        inp_matrix = tf.reshape(density_op, lindbladian_map.input_shape)
        out_matrix = lindbladian_map(inp_matrix)
        return tf.experimental.numpy.ravel(out_matrix)
    def time_deriv_jac(t: tf.Tensor, density_op: tf.Tensor) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """

        with tf.GradientTape() as tape:
            tape.watch(density_op)
            out_matrix = re_time_deriv(t, density_op)
        
        jac = tape.jacobian(out_matrix, density_op)
        return tf.experimental.numpy.ravel(jac)
    result = tfp.math.ode.BDF(rtol=rtol,atol=atol,).solve(time_deriv, 0, density_op,
                                   solution_times=[0, time], 
                                   jacobian_fn = time_deriv_jac,
                                   )
    return tf.reshape(result.states[-1], density_op.shape)

def simulate_OAT(num_qubits: int, params: tuple[float, float, float, float] | tf.Tensor, dissipation: float = 0) -> tf.Tensor:
    """
    Simulate a one-axis twisting (OAT) protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about the X axis by the angle 'params[0] * np.pi' (with Hamiltonian 'Sx').
    2. Squeeze with Hamiltonian 'Sz^2 / num_qubits' for time 'params[1] * np.pi * num_qubits'.
    3. Rotate about the axis 'X_phi' by the angle '-params[2] * np.pi',
       where 'phi = params[3] * np.pi / 2' and 'X_phi = cos(phi) X + sin(phi) Y'.

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
    assert dissipation >= 0, "dissipation cannot be negative!"

    # collective spin operators
    collective_Sx = collective_op(pauli_X, num_qubits) / 2
    collective_Sy = collective_op(pauli_Y, num_qubits) / 2
    collective_Sz = collective_op(pauli_Z, num_qubits) / 2

    if dissipation:
        # collect dissipation data as a list of tuples: (jump_rate, jump_operator)
        depolarizing_rate = dissipation / (tf.constant(numpy.pi, dtype=tf.dtypes.float64) * num_qubits)
        dissipation_data = [(depolarizing_rate, op_on_qubit(pauli / 2, qubit, num_qubits)) for pauli in [pauli_X, pauli_Y, pauli_Z] for qubit in range(num_qubits)]
    else:
        dissipation_data = []

    # initialize a state pointing down along Z (all qubits in |0>)
    state_0 = tf.Variable(tf.zeros([2**num_qubits, 2**num_qubits], dtype=tf.dtypes.complex128))
    state_0[-1, -1].assign(1)

    # rotate about the X axis
    time_0 = params[0] * tf.constant(numpy.pi, dtype=tf.dtypes.float64)
    hamiltonian_0 = collective_Sx
    state_1 = evolve_state(state_0, time_0, hamiltonian_0, *dissipation_data)
  
    # squeeze!
    time_1 = params[1] * tf.constant(numpy.pi, dtype=tf.dtypes.float64) * num_qubits
    hamiltonian_1 = collective_Sz @ collective_Sz / num_qubits
    state_2 = evolve_state(state_1, time_1, hamiltonian_1, *dissipation_data)
 
    # un-rotate about a chosen axis
    time_2 = -params[2] * tf.constant(numpy.pi, dtype=tf.dtypes.float64)
    rot_axis_angle = params[3] * tf.constant(numpy.pi, dtype=tf.dtypes.complex128) / 2
    hamiltonian_2 = tf.math.cos(rot_axis_angle) * collective_Sx + tf.math.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_2, hamiltonian_2, *dissipation_data)
    return state_3

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
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
    final_pauli_vals = [tf.math.real(tf.linalg.trace(final_state @ op)) for op in mean_ops]
    final_pauli_vars = [tf.math.real(tf.linalg.trace(final_state @ (op @ op))) - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
