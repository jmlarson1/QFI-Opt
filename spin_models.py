#!/usr/bin/env python3
import argparse
import functools
import itertools
import sys
from typing import Any, Callable, Optional, Sequence

from dissipation import Dissipator

USE_JAX = False

if USE_JAX:
    import jax

    import ode_jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as np


else:
    import numpy as np
    import scipy


COMPLEX_TYPE = np.complex128
DEFAULT_DISSIPATION_FORMAT = "XYZ"

# qubit/spin states
KET_0 = np.array([1, 0], dtype=COMPLEX_TYPE)  # |0>, spin up
KET_1 = np.array([0, 1], dtype=COMPLEX_TYPE)  # |1>, spin down

# Pauli operators
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)  # |0><0| - |1><1|
PAULI_X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)  # |0><1| + |1><0|
PAULI_Y = -1j * PAULI_Z @ PAULI_X


def log2_int(val: int) -> int:
    return val.bit_length() - 1


def simulate_sensing_protocol(
    params: Sequence[float] | np.ndarray,
    entangling_hamiltonian: np.ndarray,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
    axial_symmetry: bool = False,
) -> np.ndarray:
    """
    Simulate a sensing protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about an axis in the XY plane.
    2. Evolve under a given entangling Hamiltonian.
    3. "Un-rotate" about an axis in the XY plane.

    Step 1 rotates by an angle '+np.pi * params[0]', about the axis '2 * np.pi * params[1]'.
    Step 2 evolves under the given entangling Hamiltonian for time 'params[2] * num_qubits * np.pi'.
    Step 3 rotates by an angle '-np.pi * params[3]', about the axis '2 * np.pi * params[4]'.

    If 'axial_symmetry == True', then a 0 is inserted into the second position of 'params'.

    If dissipation_rates is nonzero, qubits experience dissipation during the entangling step (2).
    See the documentation for the Dissipator class for a general explanation of the
    dissipation_rates and dissipation_format arguments.

    This method additionally divides the dissipator (equivalently, all dissipation rates) by a
    factor of 'np.pi * num_qubits' in order to "normalize" dissipation time scales, and make them
    comparable to the time scales of coherent evolution.  Dividing a Dissipator with homogeneous
    dissipation rates 'r' by a factor of 'np.pi * num_qubits' makes so that each qubit depolarizes
    with probability 'e^(-params[2] * r)' by the end of the OAT protocol.
    """
    if axial_symmetry:
        params = np.insert(np.array(params), 1, 0.0)
    assert len(params) == 5

    num_qubits = log2_int(entangling_hamiltonian.shape[0])

    # construct collective spin operators
    collective_Sx, collective_Sy, collective_Sz = collective_spin_ops(num_qubits)

    # rotate the all-|1> state about a chosen axis
    time_1 = params[0] * np.pi
    axis_angle_1 = params[1] * 2 * np.pi
    qubit_ket = np.sin(time_1 / 2) * KET_0 + 1j * np.exp(1j * axis_angle_1) * np.cos(time_1 / 2) * KET_1
    qubit_state = np.outer(qubit_ket, qubit_ket.conj())
    state_1 = functools.reduce(np.kron, [qubit_state] * num_qubits)

    # entangle!
    time_2 = params[2] * np.pi * num_qubits
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    state_2 = evolve_state(state_1, time_2, entangling_hamiltonian, dissipator)

    # un-rotate about a chosen axis
    time_3 = -params[3] * np.pi
    axis_angle_3 = params[4] * 2 * np.pi
    final_hamiltonian = np.cos(axis_angle_3) * collective_Sx + np.sin(axis_angle_3) * collective_Sy
    state_3 = evolve_state(state_2, time_3, final_hamiltonian)

    return state_3


def simulate_OAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a one-axis twisting (OAT) protocol."""
    _, _, collective_Sz = collective_spin_ops(num_qubits)
    hamiltonian = collective_Sz.diagonal() ** 2 / num_qubits
    return simulate_sensing_protocol(params, hamiltonian, dissipation_rates, dissipation_format, axial_symmetry=True)


def simulate_TAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a two-axis twisting (TAT) protocol."""
    collective_Sx, collective_Sy, _ = collective_spin_ops(num_qubits)
    hamiltonian = (collective_Sx @ collective_Sy + collective_Sy @ collective_Sx) / num_qubits
    return simulate_sensing_protocol(params, hamiltonian, dissipation_rates, dissipation_format)


def simulate_spin_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_op: np.ndarray,
    coupling_exponent: float,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate an entangling protocol for a spin chain with power-law interactions."""
    normalization_factor = num_qubits * np.mean([1 / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)])
    hamiltonian = sum(
        act_on_subsystem(num_qubits, coupling_op, pp, qq) / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)
    )
    return simulate_sensing_protocol(params, hamiltonian / normalization_factor, dissipation_rates, dissipation_format)


def simulate_ising_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = np.kron(PAULI_Z, PAULI_Z) / 2
    return simulate_spin_chain(params, num_qubits, coupling_op, coupling_exponent, params, dissipation_rates, dissipation_format, axial_symmetry=True)


def simulate_XX_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)) / 2
    return simulate_spin_chain(coupling_op, coupling_exponent, num_qubits, params, dissipation_rates, dissipation_format, axial_symmetry=True)


def simulate_local_TAT_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_Y) + np.kron(PAULI_Y, PAULI_X)) / 2
    return simulate_spin_chain(params, num_qubits, coupling_op, coupling_exponent, dissipation_rates, dissipation_format)


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
    # treat negative times as evolving under the negative of the Hamiltonian
    # NOTE: this is required for autodiff to work
    if time.real < 0:
        time, hamiltonian = -time, -hamiltonian

    time_deriv = get_time_deriv(hamiltonian, dissipator)

    if USE_JAX:

        def _time_deriv(density_op: np.ndarray, time: float) -> np.ndarray:
            return time_deriv(time, density_op)

        times = np.linspace(0.0, time, 2)
        result = ode_jax.odeint(_time_deriv, density_op, times, rtol=rtol, atol=atol)
        return result[-1]

    else:
        density_op_shape = density_op.shape

        def _time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
            density_op.shape = density_op_shape
            output = time_deriv(time, density_op)
            density_op.shape = (-1,)
            return output.ravel()

        solution = scipy.integrate.solve_ivp(
            _time_deriv,
            [0, time.real],
            density_op.astype(complex).ravel(),
            t_eval=[time.real],
            rtol=rtol,
            atol=atol,
            method="DOP853",
        )
        final_vec = solution.y[:, -1]
        return final_vec.reshape(density_op_shape)


def get_time_deriv(hamiltonian: np.ndarray, dissipator: Optional[Dissipator] = None) -> Callable[[float, np.ndarray], np.ndarray]:
    """Construct a time derivative function that maps (time, state) --> d(state)/d(time)."""

    # construct the time derivative from coherent evolution
    if hamiltonian.ndim == 2:
        # ... with ordinary matrix multiplication
        def coherent_time_deriv(density_op: float) -> np.ndarray:
            return -1j * (hamiltonian @ density_op - density_op @ hamiltonian)

    else:
        # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
        # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
        expanded_hamiltonian = np.expand_dims(hamiltonian, 1)

        def coherent_time_deriv(density_op: float) -> np.ndarray:
            return -1j * (expanded_hamiltonian * density_op - density_op * hamiltonian)

    if not dissipator:
        return lambda time, state: coherent_time_deriv(state)
    return lambda time, state: coherent_time_deriv(state) + disipator @ state


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


def get_jacobian_func(simulate_func: Callable) -> Callable:
    """Convert a simulation method into a function that returns its Jacobian."""
    assert USE_JAX

    jacobian_func = jax.jacrev(simulate_func, argnums=(0,), holomorphic=True)

    def get_jacobian(*args: Any, **kwargs: Any) -> np.ndarray:
        return jacobian_func(*args, **kwargs)[0]

    return get_jacobian


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

    # convert the parameters into a complex array, which is necessary for autodiff capabilities
    args.params = np.array(args.params, dtype=COMPLEX_TYPE)

    if USE_JAX:
        get_jacobian = get_jacobian_func(simulate_OAT)
        jacobian = get_jacobian(args.params, args.num_qubits, args.dissipation)
        for pp in range(len(args.params)):
            print(f"d(final_state/d(params[{pp}]):")
            print(jacobian[:, :, pp])

    # simulate the OAT potocol
    final_state = simulate_OAT(args.params, args.num_qubits, args.dissipation)

    # compute collective Pauli operators
    mean_X = collective_op(PAULI_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(PAULI_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(PAULI_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_PAULI_vals = np.array([(final_state @ op).trace().real for op in mean_ops])
    final_PAULI_vars = np.array([(final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_PAULI_vals)])
    print("[<X>, <Y>, <Z>]:", final_PAULI_vals)
    print("[var(X), var(Y), var(Z)]:", final_PAULI_vars)
