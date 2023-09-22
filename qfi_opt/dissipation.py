import typing

import numpy as np
import jax.numpy as jnp


ARRAY_TYPE = typing.TypeVar("ARRAY_TYPE", np.ndarray, jnp.ndarray)


class Dissipator:
    """
    Data structure to represent a dissipation operator.
    Only (uncorrelated) single-qubit dissipation is supported at the moment.

    The Dissipator class overloads matrix multiplication, such that the time derivative of a
    density matrix 'rho' evolving only under dissipation is 'dissipator @ rho'.

    A Dissipator is initialized with the following arguments:
    (1) Three dissipation rates.  If only a single dissipation rate is provided, then all
        dissipation rates are assumed to be equal.
    (2) A string that specifies a dissipation "format".  The dissipation format may be one of the
        following:
        - "XYZ": The dissipation rates respectively correspond to dephasing along the X, Y, and Z
                 axes, with jump operators X/2, Y/2, and Z/2.  Here X, Y, and Z are Pauli operators.
                 If all dissipation rates are equal, then XYZ-type dissipation generates the
                 depolarizing channel.
        - "PMZ": The dissipation rates respectively correspond to spontaneous spin excitation,
                 decay, and dephasing, with jump operators P = (X+iY)/2, M = (X-iY)/2, and Z/2.
                 PMZ-type dissipation generates the depolarizing channel when dissipation rates are
                 proportional to (1/2, 1/2, 1).
    """

    def __init__(self, dissipation_rates: float | tuple[float, float, float], dissipation_format: str = "XYZ") -> None:
        if not isinstance(dissipation_rates, tuple):
            dissipation_rates = (dissipation_rates,) * 3
        assert all(rate >= 0 for rate in dissipation_rates), "dissipation rates cannot be negative!"
        self._bare_rates = dissipation_rates
        self._format = dissipation_format

        # This dissipator acts on a density matrix as
        #   dissipator @ rho = sum_j rate_j sum(qubit_term_j(rho, qubit) for qubit in all_qubits).
        # Identify the rates and qubit_term functions to use below.

        rates = [0.0, 0.0, 0.0]
        if dissipation_format == "XYZ":
            terms = (_qubit_term_XYZ_1, _qubit_term_XYZ_2, _qubit_term_XYZ_3)
            rate_sx, rate_sy, rate_sz = dissipation_rates
            rates[0] = (rate_sx + rate_sy) / 4 + rate_sz / 2
            rates[1] = (rate_sx + rate_sy) / 4
            rates[2] = (rate_sx - rate_sy) / 4

        elif dissipation_format == "PMZ":
            terms = (_qubit_term_PMZ_1, _qubit_term_PMZ_2, _qubit_term_PMZ_3)
            rate_sp, rate_sm, rate_sz = dissipation_rates
            rates[0] = sum(dissipation_rates) / 2
            rates[1] = rate_sp
            rates[2] = rate_sm

        else:
            raise ValueError(f"dissipation format not recognized: {dissipation_format}")

        self._rates = []
        self._terms = []
        for rate, term in zip(rates, terms):
            if rate:
                self._rates.append(rate)
                self._terms.append(term)

    def __matmul__(self, density_op: ARRAY_TYPE) -> ARRAY_TYPE | typing.Literal[0]:
        num_qubits = log2_int(density_op.size) // 2
        return sum((rate * term(density_op, num_qubits, qubit) for rate, term in zip(self._rates, self._terms) for qubit in range(num_qubits)))

    def __mul__(self, scalar: float) -> "Dissipator":
        bare_rates = (
            scalar * self._bare_rates[0],
            scalar * self._bare_rates[1],
            scalar * self._bare_rates[2],
        )
        return Dissipator(bare_rates, self._format)

    def __rmul__(self, scalar: float) -> "Dissipator":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Dissipator":
        return self * (1 / scalar)

    def __bool__(self) -> bool:
        return bool(sum(self._rates))


def log2_int(val: int) -> int:
    return val.bit_length() - 1


"""
For any specified qubit 'q', the density matrix 'rho' can be written in the form
  rho = [ [ rho_q_00, rho_q_01 ],
          [ rho_q_10, rho_q_11 ] ],
where 'rho_q_ab = <a|rho|b>_q = Tr(rho |b><a|_q)' is a block of 'rho'.

The methods below accept a density matrix and a qubit index.  These methods then organize the
density matrix into blocks (as specified above), and manipulate these blocks to construct terms
that are needed to compute the effect of dissipation.

To reduce boilerplate comments, the methods below will only specify how the blocks of the density
matrix get rearranged.
"""


def _qubit_term_XYZ_1(density_op: ARRAY_TYPE, num_qubits: int, qubit: int) -> ARRAY_TYPE:
    """
    Starting with the matrix [[a, b]]  return the matrix [[ 0, -b]
                              [c, d]],                   [[-c,  0]].
    """
    input_shape = density_op.shape
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op = density_op.reshape((dim_a, 2, dim_b, dim_a, 2, dim_b))
    if isinstance(density_op, jnp.ndarray):
        output = jnp.empty_like(density_op)
        output = output.at[:, 0, :, :, 0, :].set(0)
        output = output.at[:, 0, :, :, 1, :].set(-density_op[:, 0, :, :, 1, :])
        output = output.at[:, 1, :, :, 0, :].set(-density_op[:, 1, :, :, 0, :])
        output = output.at[:, 1, :, :, 1, :].set(0)
    else:
        output = np.empty_like(density_op)
        output[:, 0, :, :, 0, :] = 0
        output[:, 0, :, :, 1, :] = -density_op[:, 0, :, :, 1, :]
        output[:, 1, :, :, 0, :] = -density_op[:, 1, :, :, 0, :]
        output[:, 1, :, :, 1, :] = 0
    density_op = density_op.reshape(input_shape)
    return output.reshape(input_shape)


def _qubit_term_XYZ_2(density_op: ARRAY_TYPE, num_qubits: int, qubit: int) -> ARRAY_TYPE:
    """
    Starting with the matrix [[a, b]]  return the matrix [[d-a,  0 ]
                              [c, d]],                   [[ 0,  a-d]].
    """
    input_shape = density_op.shape
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op = density_op.reshape((dim_a, 2, dim_b, dim_a, 2, dim_b))
    if isinstance(density_op, jnp.ndarray):
        output = jnp.empty_like(density_op)
        output = output.at[:, 0, :, :, 0, :].set(density_op[:, 1, :, :, 1, :] - density_op[:, 0, :, :, 0, :])
        output = output.at[:, 0, :, :, 1, :].set(0)
        output = output.at[:, 1, :, :, 0, :].set(0)
        output = output.at[:, 1, :, :, 1, :].set(-output[:, 0, :, :, 0, :])
    else:
        output = np.empty_like(density_op)
        output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :] - density_op[:, 0, :, :, 0, :]
        output[:, 0, :, :, 1, :] = 0
        output[:, 1, :, :, 0, :] = 0
        output[:, 1, :, :, 1, :] = -output[:, 0, :, :, 0, :]
    density_op = density_op.reshape(input_shape)
    return output.reshape(input_shape)


def _qubit_term_XYZ_3(density_op: ARRAY_TYPE, num_qubits: int, qubit: int) -> ARRAY_TYPE:
    """
    Starting with the matrix [[a, b]]  return the matrix [[0, c]
                              [c, d]],                   [[b, 0]].
    """
    input_shape = density_op.shape
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op = density_op.reshape((dim_a, 2, dim_b, dim_a, 2, dim_b))
    if isinstance(density_op, jnp.ndarray):
        output = jnp.empty_like(density_op)
        output = output.at[:, 0, :, :, 0, :].set(0)
        output = output.at[:, 0, :, :, 1, :].set(density_op[:, 1, :, :, 0, :])
        output = output.at[:, 1, :, :, 0, :].set(density_op[:, 0, :, :, 1, :])
        output = output.at[:, 1, :, :, 1, :].set(0)
    else:
        output = np.empty_like(density_op)
        output[:, 0, :, :, 0, :] = 0
        output[:, 0, :, :, 1, :] = density_op[:, 1, :, :, 0, :]
        output[:, 1, :, :, 0, :] = density_op[:, 0, :, :, 1, :]
        output[:, 1, :, :, 1, :] = 0
    density_op = density_op.reshape(input_shape)
    return output.reshape(input_shape)


_qubit_term_PMZ_1 = _qubit_term_XYZ_1


def _qubit_term_PMZ_2(density_op: ARRAY_TYPE, num_qubits: int, qubit: int) -> ARRAY_TYPE:
    """
    Starting with the matrix [[a, b]]  return the matrix [[ d,  0]
                              [c, d]],                   [[ 0, -d]].
    """
    input_shape = density_op.shape
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op = density_op.reshape((dim_a, 2, dim_b, dim_a, 2, dim_b))
    if isinstance(density_op, jnp.ndarray):
        output = jnp.empty_like(density_op)
        output = output.at[:, 0, :, :, 0, :].set(density_op[:, 1, :, :, 1, :])
        output = output.at[:, 0, :, :, 1, :].set(0)
        output = output.at[:, 1, :, :, 0, :].set(0)
        output = output.at[:, 1, :, :, 1, :].set(-density_op[:, 1, :, :, 1, :])
    else:
        output = np.empty_like(density_op)
        output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :]
        output[:, 0, :, :, 1, :] = 0
        output[:, 1, :, :, 0, :] = 0
        output[:, 1, :, :, 1, :] = -density_op[:, 1, :, :, 1, :]
    density_op = density_op.reshape(input_shape)
    return output.reshape(input_shape)


def _qubit_term_PMZ_3(density_op: ARRAY_TYPE, num_qubits: int, qubit: int) -> ARRAY_TYPE:
    """
    Starting with the matrix [[a, b]]  return the matrix [[-a,  0]
                              [c, d]],                   [[ 0,  a]].
    """
    input_shape = density_op.shape
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op = density_op.reshape((dim_a, 2, dim_b, dim_a, 2, dim_b))
    if isinstance(density_op, jnp.ndarray):
        output = jnp.empty_like(density_op)
        output = output.at[:, 0, :, :, 0, :].set(-density_op[:, 0, :, :, 0, :])
        output = output.at[:, 0, :, :, 1, :].set(0)
        output = output.at[:, 1, :, :, 0, :].set(0)
        output = output.at[:, 1, :, :, 1, :].set(density_op[:, 0, :, :, 0, :])
    else:
        output = np.empty_like(density_op)
        output[:, 0, :, :, 0, :] = -density_op[:, 0, :, :, 0, :]
        output[:, 0, :, :, 1, :] = 0
        output[:, 1, :, :, 0, :] = 0
        output[:, 1, :, :, 1, :] = density_op[:, 0, :, :, 0, :]
    density_op = density_op.reshape(input_shape)
    return output.reshape(input_shape)
