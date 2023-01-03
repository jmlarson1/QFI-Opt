import numpy as np


def log2_int(val: int) -> int:
    return val.bit_length() - 1


class Dissipator:
    """
    Data structure for representing a dissipation operator.
    Currently only allows for single-qubit depolarizing dissipation.
    """

    def __init__(self, depolarizing_rates: float | tuple[float, float, float], dissipation_format: str = "XYZ") -> None:
        if isinstance(depolarizing_rates, float):
            depolarizing_rates = (depolarizing_rates,) * 3
        assert all(rate >= 0 for rate in depolarizing_rates), "depolarizing rates cannot be negative!"
        self._rate_sx, self._rate_sy, self._rate_sz = depolarizing_rates

        # Pauli dephasing rates differ from spin dephasing rates by a factor of 4
        self._rate_x = self._rate_sx / 4
        self._rate_y = self._rate_sy / 4
        self._rate_z = self._rate_sy / 4
        self._rate_sum = self._rate_x + self._rate_y + self._rate_z

    @property
    def is_trivial(self) -> bool:
        return self._rate_sum == 0

    def __matmul__(self, density_op: np.ndarray) -> np.ndarray:
        num_qubits = log2_int(density_op.size) // 2
        term_x = self._rate_x * sum(conjugate_by_X(density_op, qubit) for qubit in range(num_qubits))
        term_y = self._rate_y * sum(conjugate_by_Y(density_op, qubit) for qubit in range(num_qubits))
        term_z = self._rate_z * sum(conjugate_by_Z(density_op, qubit) for qubit in range(num_qubits))
        return term_x + term_y + term_z - self._rate_sum * num_qubits * density_op

    def __rmul__(self, scalar: float) -> "Dissipator":
        rates = (scalar * self._rate_sx, scalar * self._rate_sy, scalar * self._rate_sz)
        return Dissipator(rates)

    def __mul__(self, scalar: float) -> "Dissipator":
        return scalar * self

    def __truediv__(self, scalar: float) -> "Dissipator":
        return (1/scalar) * self


def conjugate_by_Z(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'Z_q rho Z_q'."""
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 0, :, :, 0, :]
    output[:, 0, :, :, 1, :] = -density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 0, :] = -density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 1, :] = density_op[:, 1, :, :, 1, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output


def conjugate_by_X(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'X_q rho X_q'."""
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :]
    output[:, 0, :, :, 1, :] = density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 0, :] = density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 1, :] = density_op[:, 0, :, :, 0, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output


def conjugate_by_Y(density_op: np.ndarray, qubit: int) -> np.ndarray:
    """For a given density operator 'rho' and qubit index 'q', return 'Y_q rho Y_q'."""
    input_shape = density_op.shape
    num_qubits = log2_int(density_op.size) // 2
    dim_a = 2**qubit
    dim_b = 2 ** (num_qubits - qubit - 1)
    density_op.shape = (dim_a, 2, dim_b, dim_a, 2, dim_b)
    output = np.empty_like(density_op)
    output[:, 0, :, :, 0, :] = density_op[:, 1, :, :, 1, :]
    output[:, 0, :, :, 1, :] = -density_op[:, 1, :, :, 0, :]
    output[:, 1, :, :, 0, :] = -density_op[:, 0, :, :, 1, :]
    output[:, 1, :, :, 1, :] = density_op[:, 0, :, :, 0, :]
    density_op.shape = input_shape
    output.shape = input_shape
    return output
