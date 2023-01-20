#!/usr/bin/env python3
import argparse
import sys

import tensorflow as tf
import tensorflow_probability as tfp
import numpy
#import numpy as np
#import scipy

# Pauli operators
pauli_Z = tf.constant([[1, 0], [0, -1]], shape=(2, 2), dtype=tf.dtypes.complex64)
pauli_X = tf.constant([[0, 1], [1, 0]], shape=(2, 2), dtype=tf.dtypes.complex64)
pauli_Y = tf.matmul(pauli_Z, pauli_X)
class LindbladianMap:
    """
    Data structure to store a Lindbladian, which is the time derivative operator for a density
    matrix undergoing Markovian time evolution.
    """

    def __init__(
        self,
        #hamiltonian: np.ndarray | scipy.sparse.spmatrix,
        #*noise_data: tuple[float, np.ndarray | scipy.sparse.spmatrix],
        #hamiltonian: tf.Tensor | tf.SparseTensor,
        #*noise_data: tuple[float, tf.Tensor | tf.SparseTensor],
        hamiltonian: tf.Tensor, 
        *noise_data: tf.Tensor,
    ) -> None:
        self._hamiltonian = hamiltonian
        #self._jump_ops = [tf.math.multiply(tf.math.sqrt(noise_rate), op) for noise_rate, op in noise_data if noise_rate]
        self._jump_ops = [tf.complex(tf.math.sqrt(noise_rate), 0.0) * op for noise_rate, op in noise_data if noise_rate]
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
    #print("op: ", op)
    #print("total_qubit_num:", total_qubit_num)
    #print("iden_before", iden_before)
    #print("iden_after", iden_after)
    inter = tf.experimental.numpy.kron(iden_before, op)
    #print("inter", inter)
    return tf.experimental.numpy.kron(inter, iden_after)
    #return tf.experimental.numpy.kron(tf.experimental.numpy.kron(iden_before, op), iden_after)


def collective_op(op: tf.Tensor, num_qubits: int) -> tf.Tensor:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return sum(op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits))


'''
#def time_deriv(_: float, density_op: tf.Tensor, lindbladian_map: LindbladianMap) -> tf.Tensor:
def time_deriv(_: float, density_op: tf.Tensor,     
                  #hamiltonian: tf.Tensor | tf.sparse.SparseTensor,
                  #noise_data: tuple[float, tf.Tensor  | tf.sparse.SparseTensor],
                  ) -> tf.Tensor:
    """
    Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
    Markovian evolution with the given Lindbladian.

    The first argument is blank to integrate with scipy.integrate.solve_ivp.
    """
    #inp_matrix = density_op.reshape(lindbladian_map.input_shape)
    #lindbladian_map = LindbladianMap(hamiltonian, *noise_data)
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
        out_matrix = time_deriv(0, density_op)
    
    jac = tape.jacobian(out_matrix, density_op)
    return tf.experimental.numpy.ravel(jac)
'''
def evolve_state(
    density_op: tf.Tensor,
    time: float,
    hamiltonian: tf.Tensor | tf.sparse.SparseTensor,
    *noise_data: tuple[float, tf.Tensor  | tf.sparse.SparseTensor],
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tf.Tensor:
    """Evolve a density operator under a Lindbladian for a given amount of time."""
    if time == 0:
        return density_op
    if time < 0:
        time, hamiltonian = -time, -hamiltonian
    #final_vec = tfp.math.ode.BDF(rtol=rtol,atol=atol,).solve(time_deriv, 0, density_op.ravel(),
    lindbladian_map = LindbladianMap(hamiltonian, *noise_data)

    def time_deriv(_: float, density_op: tf.Tensor,     
                    #hamiltonian: tf.Tensor | tf.sparse.SparseTensor,
                    #noise_data: tuple[float, tf.Tensor  | tf.sparse.SparseTensor],
                    ) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """
        #inp_matrix = density_op.reshape(lindbladian_map.input_shape)
        #lindbladian_map = LindbladianMap(hamiltonian, *noise_data)
        #print("##time_deriv: density_op ", density_op)
        #print("##time_deriv: lindbladian_map._hamiltonian ", lindbladian_map._hamiltonian)
        #print("##time_deriv: lindbladian_map._jump_ops ", lindbladian_map._jump_ops)
        #print("##time_deriv: lindbladian_map._recycling_ops ", lindbladian_map._recycling_ops)
        #print("##")
        inp_matrix = tf.reshape(density_op, lindbladian_map.input_shape)
        out_matrix = lindbladian_map(inp_matrix)
        return tf.experimental.numpy.ravel(out_matrix)

    def re_time_deriv(_: float, density_op: tf.Tensor,     
                    #hamiltonian: tf.Tensor | tf.sparse.SparseTensor,
                    #noise_data: tuple[float, tf.Tensor  | tf.sparse.SparseTensor],
                    ) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """
        #inp_matrix = density_op.reshape(lindbladian_map.input_shape)
        #lindbladian_map = LindbladianMap(hamiltonian, *noise_data)
        #print("##RE time_deriv: density_op ", density_op)
        #print("##RE time_deriv: lindbladian_map._hamiltonian ", lindbladian_map._hamiltonian)
        #print("##RE time_deriv: lindbladian_map._jump_ops ", lindbladian_map._jump_ops)
        #print("##RE time_deriv: lindbladian_map._recycling_ops ", lindbladian_map._recycling_ops)
        #print("##RE ")
        inp_matrix = tf.reshape(density_op, lindbladian_map.input_shape)
        out_matrix = lindbladian_map(inp_matrix)
        return tf.experimental.numpy.ravel(out_matrix)
    def time_deriv_jac(t: tf.Tensor, density_op: tf.Tensor) -> tf.Tensor:
        """
        Compute the time derivative of the given density operator (flattened to a 1D vector) undergoing
        Markovian evolution with the given Lindbladian.

        The first argument is blank to integrate with scipy.integrate.solve_ivp.
        """

        #print("time_deriv_jac density_op", density_op)
        with tf.GradientTape() as tape:
            tape.watch(density_op)
            out_matrix = re_time_deriv(t, density_op)
        
        jac = tape.jacobian(out_matrix, density_op)
        #print("JAC", jac)
        return tf.experimental.numpy.ravel(jac)
    print("outer density_op ", density_op)
    print("outer hamiltonian ", hamiltonian)
    print("outer noise_data ", noise_data)
    result = tfp.math.ode.BDF(rtol=rtol,atol=atol,).solve(time_deriv, 0, density_op,
                                   solution_times=[0, time], 
                                   #constants={'hamiltonian': hamiltonian, 
                                   #           'noise_data': noise_data},
                                   jacobian_fn = time_deriv_jac,
                                   #constants={'LindbladianMap': 
                                   # LindbladianMap(hamiltonian, *noise_data)}
                                   )
    #print("result ", result)
    #print("result.states[-1] ", result.states[-1])
    return tf.reshape(result.states[-1], density_op.shape)

def simulate_OAT(num_qubits: int, params: tuple[float, float, float, float] | tf.Tensor, noise_level: float = 0) -> tf.Tensor:
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
        depolarizing_rate = noise_level / (tf.constant(numpy.pi, dtype=tf.dtypes.float32) * num_qubits)
        noise_data = [(depolarizing_rate, op_on_qubit(pauli / 2, qubit, num_qubits)) for pauli in [pauli_X, pauli_Y, pauli_Z] for qubit in range(num_qubits)]
    else:
        noise_data = []

    #print("noise_data", noise_data)

    # initialize a state pointing down along Z (all qubits in |0>)
    #SHK. PK to review shape
    state_0 = tf.Variable(tf.zeros([2**num_qubits, 2**num_qubits], dtype=tf.dtypes.complex64))
    #state_0 = tf.zeros((2**num_qubits,) * 2, dtype=tf.dtypes.complex64)
    state_0[-1, -1].assign(1)

    # rotate about the X axis
    time_0 = params[0] * tf.constant(numpy.pi, dtype=tf.dtypes.float32)
    #print("time_0", time_0)
    #print("started state_1 ")
    hamiltonian_0 = collective_Sx
    #print("hamiltonian_0 ", hamiltonian_0)
    state_1 = evolve_state(state_0, time_0, hamiltonian_0, *noise_data)
    #print("computed state_1 ")
    # squeeze!
    time_1 = params[1] * tf.constant(numpy.pi, dtype=tf.dtypes.float32) * num_qubits
    hamiltonian_1 = collective_Sz @ collective_Sz / num_qubits
    state_2 = evolve_state(state_1, time_1, hamiltonian_1, *noise_data)
    #print("computed state_2 ")
    # un-rotate about a chosen axis
    time_2 = -params[2] * tf.constant(numpy.pi, dtype=tf.dtypes.float32)
    rot_axis_angle = params[3] * tf.constant(numpy.pi, dtype=tf.dtypes.complex64) / 2
    hamiltonian_2 = tf.math.cos(rot_axis_angle) * collective_Sx + tf.math.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_2, hamiltonian_2, *noise_data)
    #print("computed state_3 ")
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
    final_pauli_vals = [tf.math.real(tf.linalg.trace(final_state @ op)) for op in mean_ops]
    final_pauli_vars = [tf.math.real(tf.linalg.trace(final_state @ (op @ op))) - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
