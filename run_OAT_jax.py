import argparse
import jax
import functools
import sys
#
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax._src.public_test_util import check_grads as check_grads
from dissipation_jax import Dissipator
from ode_jax import odeint

COMPLEX_DTYPE = jnp.complex128

# qubit/spin states
ket_0 = jnp.array([1, 0], dtype=COMPLEX_DTYPE)  # |0>, spin up
ket_1 = jnp.array([0, 1], dtype=COMPLEX_DTYPE)  # |1>, spin down

# Pauli operators
pauli_Z = jnp.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)  # |0><0| - |1><1|
pauli_X = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)  # |0><1| + |1><0|
pauli_Y = -1j * pauli_Z @ pauli_X


def simulate_OAT(
    params: jnp.ndarray,
    num_qubits: int = 4,
    dissipation_rates: float = 0.0,
    dissipation_format: str = "XYZ",
) -> jnp.ndarray:
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

    # construct collective spin operators
    collective_Sx, collective_Sy, collective_Sz = collective_spin_ops(num_qubits)
    # rotate the all-|1> state about the X axis
    time_1 = params[0] * jnp.pi
    qubit_ket = jnp.sin(time_1 / 2) * ket_0 + 1j * jnp.cos(time_1 / 2) * ket_1
    qubit_state = jnp.outer(qubit_ket, jnp.conj(qubit_ket))
    state_1 = functools.reduce(jnp.kron, [qubit_state] * num_qubits)

    # squeeze!
    time_2 = params[1] * jnp.pi * num_qubits
    hamiltonian_2 = jnp.diagonal(collective_Sz) ** 2 / num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (jnp.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, hamiltonian_2, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[2] * jnp.pi
    rot_axis_angle = params[3] * 2 * jnp.pi
    hamiltonian_3 = jnp.cos(rot_axis_angle) * collective_Sx + jnp.sin(rot_axis_angle) * collective_Sy
    state_3 = evolve_state(state_2, time_3, hamiltonian_3)

    return state_3

def evolve_state(
    density_op_arg,
    time: float,
    hamiltonian_arg,
    dissipator: Dissipator = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
):
    """
    Time-evolve a given initial density operator for a given amount of time under the given Hamiltonian and (optionally) Dissipator.
    """
    if jnp.real(time) == 0:
        return density_op_arg
    if jnp.real(time) < 0:
        time, hamiltonian_arg = -time, -hamiltonian_arg

    def time_deriv(density_op, _: float, args):
        #return dissipator @ density_op
        # compute commutator with hamiltonian
        hamiltonian = args[0]
        if hamiltonian.ndim == 2:
            # ... computed with ordinary matrix multiplication
            output = -1j * (hamiltonian @ density_op - density_op @ hamiltonian)
        else:
            # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
            # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
            output = -1j * (jnp.expand_dims(hamiltonian, 1) * density_op - density_op * hamiltonian)
        if dissipator is None:
            return output
        return output + dissipator @ density_op

    #TODO: Decide the number of timesteps
    t = jnp.linspace(0., time, 100)
    result = odeint(time_deriv, density_op_arg, t, (hamiltonian_arg,), rtol=rtol, atol=atol, mxstep=jnp.inf, hmax=jnp.inf)
    return result[-1]


@functools.cache
def collective_spin_ops(num_qubits: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct collective spin operators."""
    return (
        collective_op(pauli_X, num_qubits) / 2,
        collective_op(pauli_Y, num_qubits) / 2,
        collective_op(pauli_Z, num_qubits) / 2,
    )


def collective_op(op: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
    """Compute the collective version of a qubit operator: sum_q op_q."""
    return jnp.array(sum(
        (op_on_qubit(op, qubit, num_qubits) for qubit in range(num_qubits)),
        start=0))


def op_on_qubit(op: jnp.ndarray, qubit: int, total_qubit_num: int) -> jnp.ndarray:
    """
    Return an operator that acts with 'op' in the given qubit, and trivially (with the identity operator) on all other qubits.
    """
    iden_before = jnp.eye(2**qubit, dtype=op.dtype)
    iden_after = jnp.eye(2 ** (total_qubit_num - qubit - 1), dtype=op.dtype)
    #return jnp.kron(jnp.kron(iden_before, op), iden_after)
    return functools.reduce(jnp.kron, [iden_before, op, iden_after])

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

    params_jax = jnp.array(args.params, dtype=COMPLEX_DTYPE)
    jacrev_fun = jax.jacrev(simulate_OAT, argnums=(0,),holomorphic=True)

    #check_grads(simulate_OAT, (params_jax,), 1,  modes=("rev"))
    jac = jacrev_fun(params_jax, args.num_qubits, args.dissipation)

    for i in range(jac[0].shape[0]):
        for j in range(jac[0].shape[1]):
            print("d(finalstate[",i,",",j,"])/d(params)= ", jac[0][i][j])

    final_state = simulate_OAT(params_jax, args.num_qubits, args.dissipation)
    # compute collective Pauli operators
    mean_X = collective_op(pauli_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(pauli_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(pauli_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_pauli_vals = [jnp.real(jnp.trace(final_state @ op)) for op in mean_ops]
    final_pauli_vars = [
    jnp.real(jnp.trace(final_state @ (op @ op))) - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)
]
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
