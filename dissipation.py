from typing import Literal

import numpy as np


def log2_int(val: int) -> int:
    return val.bit_length() - 1


class Dissipator:
    """
    Data structure that represents a dissipation operator.
    """

    def __init__(self, dissipation_rates: float | tuple[float, float, float], dissipation_format: str = "XYZ") -> None:
        if isinstance(dissipation_rates, float):
            dissipation_rates = (dissipation_rates,) * 3
        assert all(rate >= 0 for rate in dissipation_rates), "dissipation rates cannot be negative!"
        self._rates = dissipation_rates
        self._format = dissipation_format

        if dissipation_format == "XYZ":
            rate_sx, rate_sy, rate_sz = dissipation_rates
            self._rate_1 = (rate_sx + rate_sy) / 4 + rate_sz / 2
            self._rate_2 = (rate_sx + rate_sy) / 4
            self._rate_3 = (rate_sx - rate_sy) / 4
            self._qubit_term_1 = _qubit_term_XYZ_1
            self._qubit_term_2 = _qubit_term_XYZ_2
            self._qubit_term_3 = _qubit_term_XYZ_3

        elif dissipation_format == "PMZ":
            rate_sp, rate_sm, rate_sz = dissipation_rates
            self._rate_1 = sum(dissipation_rates) / 2
            self._rate_2 = rate_sp
            self._rate_3 = rate_sm
            self._qubit_term_1 = _qubit_term_PMZ_1
            self._qubit_term_2 = _qubit_term_PMZ_2
            self._qubit_term_3 = _qubit_term_PMZ_3

        else:
            raise ValueError(f"dissipation format not recognized {dissipation_format}")

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
        return Dissipator(rates, self._format)

    def __mul__(self, scalar: float) -> "Dissipator":
        return scalar * self

    def __truediv__(self, scalar: float) -> "Dissipator":
        return (1 / scalar) * self

    @property
    def is_trivial(self) -> bool:
        return sum(self._rates) == 0


def _qubit_term_XYZ_1(density_op: np.ndarray, qubit: int) -> np.ndarray:
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
