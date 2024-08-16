#!/usr/bin/env python3
import argparse
import functools
import itertools
import os
import sys
from collections.abc import Callable, Sequence

import diffrax
import numpy

from qfi_opt.dissipation import Dissipator

USE_DIFFRAX = bool(os.getenv("USE_DIFFRAX"))
FORWARD_MODE = not bool(os.getenv("REVERSE_MODE"))

if USE_DIFFRAX:
    import jax
    import jax.numpy as np

    jax.config.update("jax_enable_x64", True)

else:
    import numpy as np  # type: ignore[no-redef]
    import scipy

COMPLEX_TYPE = np.complex128
DEFAULT_DISSIPATION_FORMAT = "XYZ"

# qubit/spin states
KET_0 = np.array([1, 0], dtype=COMPLEX_TYPE)  # |0>, spin up
KET_1 = np.array([0, 1], dtype=COMPLEX_TYPE)  # |1>, spin down

# Pauli operators
PAULI_I = np.array([[1, 0], [0, 1]], dtype=COMPLEX_TYPE)  # |0><0| + |1><1|
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=COMPLEX_TYPE)  # |0><0| - |1><1|
PAULI_X = np.array([[0, 1], [1, 0]], dtype=COMPLEX_TYPE)  # |0><1| + |1><0|
PAULI_Y = -1j * PAULI_Z @ PAULI_X


def log2_int(val: int) -> int:
    return val.bit_length() - 1


def simulate_sensing_protocol(
    params: Sequence[float] | np.ndarray,
    entangling_hamiltonian: np.ndarray,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
    axial_symmetry: bool = False,
) -> np.ndarray:
    """Simulate a sensing protocol, and return the final state (density matrix).

    Starting with an initial all-|1> state (all spins pointing down along the Z axis):
    1. Rotate about an axis in the XY plane.
    2. Evolve under a given entangling Hamiltonian.
    3. Rotate about the X axis.
    4. Rotate about the Y axis.

    Step 1 rotates by an angle 'np.pi * params[0]', about the axis 'np.pi * params[1]'.
    Step 2 evolves under the given entangling Hamiltonian for time 'params[2] * np.pi * num_qubits'.
    Step 3 rotates by an angle 'np.pi * params[3]'.
    Step 3 rotates by an angle 'np.pi * params[-1]'.

    If axial_symmetry is set to True, 0 is appended to the parameter list.
    Afterwards, if there are more than 5 parameters, steps 2 and 3 are repeated as appropriate.

    If dissipation_rates is nonzero, qubits experience dissipation during the entangling steps.
    See the documentation for the Dissipator class for a general explanation of the
    dissipation_rates and dissipation_format arguments.

    This method additionally divides the dissipator (equivalently, all dissipation rates) by a
    factor of 'np.pi * num_qubits' in order to "normalize" dissipation time scales, and make them
    comparable to the time scales of coherent evolution.  Dividing a Dissipator with homogeneous
    dissipation rates 'r' by a factor of 'np.pi * num_qubits' makes so that each qubit depolarizes
    with probability 'e^(-params[0] * r)' by the end of the OAT protocol.
    """
    if len(params) < 5 or not len(params) % 2:
        raise ValueError(f"The number of parameters should be an odd number >=5, not {len(params)}.")

    num_qubits = log2_int(entangling_hamiltonian.shape[0])

    # rotate the all-|1> state about a chosen axis
    time = params[0] * np.pi
    cos = np.cos(time / 2)
    sin = np.sin(time / 2)
    axis_angle = params[1] * np.pi
    qubit_ket = cos * KET_1 - 1j * np.exp(-1j * axis_angle) * sin * KET_0
    qubit_state = np.outer(qubit_ket, qubit_ket.conj())
    state = functools.reduce(np.kron, [qubit_state] * num_qubits)

    # entangle - rotate layers
    dissipator = Dissipator(dissipation_rates, dissipation_format) / (np.pi * num_qubits)
    for pp in range(2, len(params) - 1, 2):
        # entangle
        time = params[pp] * np.pi * num_qubits
        state = evolve_state(state, time, entangling_hamiltonian, dissipator)

        # rotate about Sx
        time = params[pp + 1] * np.pi
        mat_rx = np.cos(time / 2) * PAULI_I - 1j * np.sin(time / 2) * PAULI_X
        state = apply_globally(state, mat_rx, num_qubits)

    # rotate about Sy
    time = params[-1] * np.pi
    mat_ry = np.cos(time / 2) * PAULI_I - 1j * np.sin(time / 2) * PAULI_Y
    state = apply_globally(state, mat_ry, num_qubits)

    return state


def apply_globally(density_op: np.ndarray, qubit_op: np.ndarray, num_qubits: int) -> np.ndarray:
    """Apply the given qubit operator to all qubits of a density operator."""
    qubit_op_dag = qubit_op.conj().T
    for qubit in range(num_qubits):
        dim_a = 2**qubit
        dim_b = 2 ** (num_qubits - qubit - 1)
        density_op = density_op.reshape((dim_a, 2, dim_b * dim_a, 2, dim_b))
        density_op = np.einsum("ij,AjBkC,kl->AiBlC", qubit_op, density_op, qubit_op_dag)
    return density_op.reshape((2**num_qubits,) * 2)


def enable_axial_symmetry(simulate_func: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    """Decorator to enable an axially-symmetric version of a simulation method.

    Axial symmetry means that the last parameter can be set to zero without loss of generality.
    """

    def simulate_func_with_symmetry(params: Sequence[float] | np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        if len(params) == 4:
            # Verify that the dissipation arguments are compatible with axial symmetry.
            dissipation_rates = kwargs.get("dissipation_rates", 0.0)
            dissipation_format = kwargs.get("dissipation_format", DEFAULT_DISSIPATION_FORMAT)
            if dissipation_format == "XYZ" and hasattr(dissipation_rates, "__iter__"):
                rate_sx, rate_sy, *_ = dissipation_rates
                if not rate_sx == rate_sy:
                    raise ValueError(
                        f"Dissipation format {dissipation_format} with rates {dissipation_rates} does not respect axial symmetry."
                        "\nPlease provide at least 5 parameters, or pick different options."
                    )
            params = np.append(np.array(params), 0.0)

        return simulate_func(params, *args, **kwargs)

    return simulate_func_with_symmetry


@enable_axial_symmetry
def simulate_OAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a one-axis twisting (OAT) protocol."""
    _, _, collective_Sz = collective_spin_ops(num_qubits)
    hamiltonian = collective_Sz.diagonal() ** 2 / num_qubits
    return simulate_sensing_protocol(
        params,
        hamiltonian,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_TAT(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate a two-axis twisting (TAT) protocol."""
    collective_Sx, collective_Sy, _ = collective_spin_ops(num_qubits)
    hamiltonian = (collective_Sx @ collective_Sy + collective_Sy @ collective_Sx) / num_qubits
    return simulate_sensing_protocol(
        params,
        hamiltonian,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_spin_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_op: np.ndarray,
    coupling_exponent: float = 1.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    """Simulate an entangling protocol for a spin chain with power-law interactions."""
    normalization_factor = num_qubits * np.array([1 / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)]).mean()
    hamiltonian = sum(
        act_on_subsystem(num_qubits, coupling_op, pp, qq) / abs(pp - qq) ** coupling_exponent for pp, qq in itertools.combinations(range(num_qubits), 2)
    )
    return simulate_sensing_protocol(
        params,
        hamiltonian / normalization_factor,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


@enable_axial_symmetry
def simulate_ising_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 1.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = np.kron(PAULI_Z, PAULI_Z) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


@enable_axial_symmetry
def simulate_XX_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 1.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_X) + np.kron(PAULI_Y, PAULI_Y)) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def simulate_local_TAT_chain(
    params: Sequence[float] | np.ndarray,
    num_qubits: int,
    coupling_exponent: float = 1.0,
    *,
    dissipation_rates: float | tuple[float, float, float] = 0.0,
    dissipation_format: str = DEFAULT_DISSIPATION_FORMAT,
) -> np.ndarray:
    coupling_op = (np.kron(PAULI_X, PAULI_Y) + np.kron(PAULI_Y, PAULI_X)) / 2
    return simulate_spin_chain(
        params,
        num_qubits,
        coupling_op,
        coupling_exponent,
        dissipation_rates=dissipation_rates,
        dissipation_format=dissipation_format,
    )


def evolve_state(
    density_op: np.ndarray,
    time: float | np.ndarray,
    hamiltonian: np.ndarray,
    dissipator: Dissipator | None = None,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    solver: diffrax.AbstractSolver | None = None,
    **diffrax_kwargs: object,
) -> np.ndarray:
    """Time-evolve a given initial density operator for a given amount of time under the given Hamiltonian and (optionally) Dissipator."""

    # treat negative times as evolving under the negative of the Hamiltonian
    # NOTE: this is required for autodiff to work
    if time.real < 0:
        time, hamiltonian = -time, -hamiltonian

    time_deriv = get_time_deriv(hamiltonian, dissipator)

    if USE_DIFFRAX:

        # set initial time step size, if necessary
        if "dt0" not in diffrax_kwargs:
            diffrax_kwargs["dt0"] = 0.002

        def _time_deriv(time: float, density_op: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
            return time_deriv(time, density_op)

        term: diffrax.ODETerm = diffrax.ODETerm(_time_deriv)
        solver = solver or diffrax.Tsit5()  # try also diffrax.Dopri8()
        solver_args: dict[str, object] = dict(t0=0.0, t1=time.real, y0=density_op, args=(hamiltonian,))
        if FORWARD_MODE:
            diffrax_kwargs["max_steps"] = diffrax_kwargs.get("max_steps", None)
            solver_args |= dict(adjoint=diffrax.DirectAdjoint())
        solution = diffrax.diffeqsolve(term, solver, **solver_args, **diffrax_kwargs)
        return solution.ys[-1]

    else:
        if np.isclose(time, 0, atol=atol):
            return density_op

        def scipy_time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
            density_op.shape = (hamiltonian.shape[0],) * 2  # type: ignore[misc]
            output = time_deriv(time, density_op)
            density_op.shape = (-1,)  # type: ignore[misc]
            return output.ravel()

        result = scipy.integrate.solve_ivp(
            scipy_time_deriv,
            [0, time],
            density_op.ravel(),
            t_eval=[time],
            rtol=rtol,
            atol=atol,
        )
        return result.y[:, -1].reshape(density_op.shape)


def get_time_deriv(
    hamiltonian: np.ndarray,
    dissipator: Dissipator | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Construct a time derivative function that maps (state, time) --> d(state)/d(time)."""

    # construct the time derivative from coherent evolution
    if hamiltonian.ndim == 2:
        # ... with ordinary matrix multiplication
        def coherent_time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
            return -1j * (hamiltonian @ density_op - density_op @ hamiltonian)

    else:
        # 'hamiltonian' is a 1-D array of the values on the diagonal of the actual Hamiltonian,
        # so we can compute the commutator with array broadcasting, which is faster than matrix multiplication
        expanded_hamiltonian = np.expand_dims(hamiltonian, 1)

        def coherent_time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
            return -1j * (expanded_hamiltonian * density_op - density_op * hamiltonian)

    if not dissipator:
        return coherent_time_deriv

    def dissipative_time_deriv(time: float, density_op: np.ndarray) -> np.ndarray:
        return coherent_time_deriv(time, density_op) + dissipator @ density_op

    return dissipative_time_deriv


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
    """Return an operator that acts with 'op' in the given qubits, and trivially (with the identity operator) on all other qubits."""
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

    if USE_DIFFRAX and FORWARD_MODE:
        # forward-mode automatic differentiation

        def get_jacobian(params: Sequence[float], *args: object, **kwargs: object) -> np.ndarray:
            _simulate_func = lambda params: simulate_func(params, *args, **kwargs)
            _get_jacobian = jax.jacfwd(_simulate_func, argnums=0, holomorphic=True)
            return _get_jacobian(np.array(params, dtype=COMPLEX_TYPE))

        return get_jacobian

    elif USE_DIFFRAX and not FORWARD_MODE:
        # reverse-mode automatic differentiation

        def get_jacobian(params: Sequence[float], *args: object, **kwargs: object) -> np.ndarray:
            primals, vjp_func = jax.vjp(simulate_func, params, *args)
            result = np.zeros((primals.shape[1], primals.shape[0], len(params)), dtype=COMPLEX_TYPE)
            for ii, jj in numpy.ndindex(primals.shape):
                seed = np.zeros(primals.shape, dtype=COMPLEX_TYPE)
                seed = seed.at[ii, jj].set(1.0)
                real_part = np.array(vjp_func(seed)[0]).flatten()
                seed = seed.at[ii, jj].set(1.0j)
                imag_part = np.array(vjp_func(seed)[0]).flatten()
                # Take the conjugate to account for Jax convention. See discussion:
                # https://github.com/google/jax/issues/4891
                result = result.at[ii, jj, :].set(real_part - 1.0j * imag_part)
            return result

        return get_jacobian

    def get_jacobian_manually(params: Sequence[float], *args: object, **kwargs: object) -> np.ndarray:
        step_sizes = kwargs.get("step_sizes", 1e-10)
        if isinstance(step_sizes, float):
            param_step_sizes = [step_sizes] * len(params)
        assert len(param_step_sizes) == len(params)

        result_at_params = simulate_func(params, *args, **kwargs)
        shifted_results = []
        for idx, step_size in enumerate(param_step_sizes):
            new_params = list(params)
            new_params[idx] += step_size
            result_at_params_with_step = simulate_func(new_params, *args, **kwargs)
            shifted_results.append((result_at_params_with_step - result_at_params) / step_size)

        return np.stack(shifted_results, axis=-1)

    return get_jacobian_manually


def print_jacobian(jacobian: np.ndarray, precision: int = 3, linewidth: int = 200) -> None:
    np.set_printoptions(precision=precision, suppress=True, linewidth=linewidth)
    params = jacobian.shape[2]
    for pp in range(params):
        print(f"d(final_state/d(params[{pp}]):")
        print(jacobian[:, :, pp])


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Simulate a simple one-axis twisting (OAT) protocol.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_qubits", type=int, default=4)
    parser.add_argument("--dissipation", type=float, default=0.0)
    parser.add_argument("--params", type=float, nargs="+", default=np.array([0.5, 0.5, 0.5, 0]))
    parser.add_argument("--jacobian", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.jacobian:
        get_jacobian = get_jacobian_func(simulate_OAT)
        jacobian = get_jacobian(args.params, args.num_qubits, dissipation_rates=args.dissipation)
        print_jacobian(jacobian)

    # simulate the OAT protocol
    final_state = simulate_OAT(args.params, args.num_qubits, dissipation_rates=args.dissipation)

    # compute collective Pauli operators
    mean_X = collective_op(PAULI_X, args.num_qubits) / args.num_qubits
    mean_Y = collective_op(PAULI_Y, args.num_qubits) / args.num_qubits
    mean_Z = collective_op(PAULI_Z, args.num_qubits) / args.num_qubits
    mean_ops = [mean_X, mean_Y, mean_Z]

    # print out expectation values and variances
    final_pauli_vals = np.array([(final_state @ op).trace().real for op in mean_ops])
    final_pauli_vars = np.array([(final_state @ (op @ op)).trace().real - mean_op_val**2 for op, mean_op_val in zip(mean_ops, final_pauli_vals)])
    print("[<X>, <Y>, <Z>]:", final_pauli_vals)
    print("[var(X), var(Y), var(Z)]:", final_pauli_vars)
