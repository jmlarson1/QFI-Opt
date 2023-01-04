import numpy as np


def log2_int(val: int) -> int:
    return val.bit_length() - 1


class Dissipator:
    """
    Data structure to represent a dissipation operator.
    Only (uncorrelated) single-qubit dissipation is supported at the moment.

    A Dissipator is initialized with the following arguments:
    (1) Three dissipation rates.  If only a single dissipation rate is provided, then all
        dissipation rates are assumed to be equal.
    (2) A string that specifies a dissipation "type".  The dissipation type may be one of the
        following:
        - "XYZ": The dissipation rates respectively correspond to dephasing along the X, Y, and Z
                 axes, with jump operators X/2, Y/2, Z/2.  Here X, Y, and Z are Pauli operators.
        - "XYZ": The dissipation rates respectively correspond to spontaneous spin excitation,
                 relaxation, and dephasing, with jump operators P = (X+iY)/2, M = (X-iY)/2, and Z/2.

    The Dissipator class overloads the matrix multiplication operator, such that the time derivative
    of a density matrix undergoing only dissipation (and no coherent evolution) is 'dissipator @ rho'.
    """

    def __init__(self, dissipation_rates: float | tuple[float, float, float], dissipation_type: str = "XYZ") -> None:
        if isinstance(dissipation_rates, float):
            dissipation_rates = (dissipation_rates,) * 3
        assert all(rate >= 0 for rate in dissipation_rates), "dissipation rates cannot be negative!"
        self._rates = dissipation_rates
        self._dissipation_type = dissipation_type

        if dissipation_type == "XYZ":
            rate_sx, rate_sy, rate_sz = dissipation_rates
            self._rate_1 = (rate_sx + rate_sy) / 4 + rate_sz / 2
            self._rate_2 = (rate_sx + rate_sy) / 4
            self._rate_3 = (rate_sx - rate_sy) / 4
            self._qubit_term_1 = _qubit_term_XYZ_1
            self._qubit_term_2 = _qubit_term_XYZ_2
            self._qubit_term_3 = _qubit_term_XYZ_3

        elif dissipation_type == "PMZ":
            rate_sp, rate_sm, rate_sz = dissipation_rates
            self._rate_1 = sum(dissipation_rates) / 2
            self._rate_2 = rate_sp
            self._rate_3 = rate_sm
            self._qubit_term_1 = _qubit_term_PMZ_1
            self._qubit_term_2 = _qubit_term_PMZ_2
            self._qubit_term_3 = _qubit_term_PMZ_3

        else:
            raise ValueError(f"dissipation format not recognized {dissipation_type}")

    def __matmul__(self, density_op: np.ndarray) -> np.ndarray | float:
        num_qubits = log2_int(density_op.size) // 2
        output = self._rate_1 * sum(self._qubit_term_1(density_op, qubit) for qubit in range(num_qubits))
        if self._rate_2:
            output += self._rate_2 * sum(self._qubit_term_2(density_op, qubit) for qubit in range(num_qubits))
        if self._rate_3:
            output += self._rate_3 * sum(self._qubit_term_3(density_op, qubit) for qubit in range(num_qubits))
        return output

    def __rmul__(self, scalar: float) -> "Dissipator":
        rates = (scalar * self._rates[0], scalar * self._rates[1], scalar * self._rates[2])
        return Dissipator(rates, self._dissipation_type)

    def __mul__(self, scalar: float) -> "Dissipator":
        return scalar * self

    def __truediv__(self, scalar: float) -> "Dissipator":
        return (1 / scalar) * self

    @property
    def is_trivial(self) -> bool:
        return sum(self._rates) == 0


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


def _qubit_term_XYZ_1(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """
    Starting with the matrix [[a, b]]  return the matrix [[ 0, -b]
                              [c, d]],                   [[-c,  0]].
    """
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = 0
    output[:, 0, :, :, 1, :] = -density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 0, :] = -density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 1, :] = 0
    density_op.shape = input_shape
    output.shape = input_shape
    return output


def _qubit_term_XYZ_2(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """
    Starting with the matrix [[a, b]]  return the matrix [[d-a,  0 ]
                              [c, d]],                   [[ 0,  a-d]].
    """
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :] - density_op[:, 0, :, :, 0, :]
    output[:, 0, :, :, 1, :] = 0
    output[:, 1, :, :, 0, :] = 0
    output[:, 1, :, :, 1, :] = -output[:, 0, :, :, 0, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output


def _qubit_term_XYZ_3(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """
    Starting with the matrix [[a, b]]  return the matrix [[0, c]
                              [c, d]],                   [[b, 0]].
    """
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = 0
    output[:, 0, :, :, 1, :] = density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 0, :] = density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 1, :] = 0
    density_op.shape = input_shape
    output.shape = input_shape
    return output


_qubit_term_PMZ_1 = _qubit_term_XYZ_1


def _qubit_term_PMZ_2(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """
    Starting with the matrix [[a, b]]  return the matrix [[ d,  0]
                              [c, d]],                   [[ 0, -d]].
    """
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :]
    output[:, 0, :, :, 1, :] = 0
    output[:, 1, :, :, 0, :] = 0
    output[:, 1, :, :, 1, :] = -density_op[:, 1, :, :, 1, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output


def _qubit_term_PMZ_3(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """
    Starting with the matrix [[a, b]]  return the matrix [[-a,  0]
                              [c, d]],                   [[ 0,  a]].
    """
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = -density_op[:, 0, :, :, 0, :]
    output[:, 0, :, :, 1, :] = 0
    output[:, 1, :, :, 0, :] = 0
    output[:, 1, :, :, 1, :] = density_op[:, 0, :, :, 0, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output
