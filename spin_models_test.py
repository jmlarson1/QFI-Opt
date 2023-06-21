import dataclasses
import functools
from typing import Callable, List, Optional

import jax.numpy as np
import numpy

import spin_models

OATParams = tuple[float, float, float, float]


@dataclasses.dataclass(kw_only=True)
class Transformation:
    """An object representing a sequence of transformations of a quantum state."""

    flip_z: bool = False
    flip_xy: Optional[float] = None
    conjugate: bool = False

    def transform(self, state: np.ndarray) -> np.ndarray:
        num_qubits = int(np.log2(state.shape[0]))
        new_state = state.copy()
        if self.flip_z:
            # apply the global spin rotation 'Rz(pi)'
            new_state = rot_z_mat(num_qubits, np.pi) * new_state
        if self.flip_xy is not None:
            # apply a global spin rotation by an angle 'pi' about an axis in the X-Y plane
            phi = 2 * np.pi * self.flip_xy
            phase_mat = rot_z_mat(num_qubits, phi)
            new_state = phase_mat * (phase_mat.conj() * new_state)[::-1, ::-1]
        if self.conjugate:
            # complex conjugate the state
            new_state = new_state.conj()
        return new_state


@functools.cache
def rot_z_mat(num_qubits: int, angle: float) -> np.ndarray:
    """
    Construct the matrix 'phase_mat' for which element-wise multiplication 'phase_mat * density_op' rotates the state 'density_op' about the Z axis by the
    given angle.
    """
    if not angle:
        return np.ones((2**num_qubits,) * 2)
    _, _, collective_Sz = spin_models.collective_spin_ops(num_qubits)
    phase_vec = np.exp(-1j * angle * collective_Sz.diagonal())
    return phase_vec * np.conj(phase_vec[:, np.newaxis])


def get_symmetries(num_qubits: int) -> List[Callable[[float, float, float, float], tuple[OATParams, Transformation]]]:
    """
    Generate a list of symmetries of the OAT protocol at zero dissipation.

    Each symmetry is a map from 'old_params' --> '(new_params, transformation)', where 'transformation' indicates how a quantum state prepared with the
    'new_params' should be additionally transformed to recover an exact symmetry.  Note that the additional transformations (should) have no effect on the QFI.
    """
    even_qubit_number = num_qubits % 2 == 0

    def shift_1(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1 + 1, t_OAT, t_2, -aa), Transformation(flip_xy=0)

    def shift_2(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT, t_2 + 1, aa), Transformation(flip_xy=-aa)

    def reflect_1(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (-t_1, t_OAT, t_2, aa + 0.5), Transformation(flip_z=True)

    def reflect_2(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT, -t_2, aa + 0.5), Transformation()

    def shift_OAT(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT + 1, t_2, aa + 0.5 * even_qubit_number), Transformation(flip_z=even_qubit_number)

    def reflect_OAT(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, -t_OAT, t_2, -aa), Transformation(flip_z=True, conjugate=True)

    return [shift_1, shift_2, reflect_1, reflect_2, shift_OAT, reflect_OAT]


def test_symmetries() -> None:
    """Test the symmetry transformations that we used to cut down the domain of the parameters for the OAT protocol."""
    for _ in range(10):  # test several random parameters
        params = numpy.random.random(4)
        for num_qubits in [2, 3]:  # test both even and odd qubit numbers
            state = spin_models.simulate_OAT(params, num_qubits)
            for symmetry in get_symmetries(num_qubits):
                new_params, transformation = symmetry(*params)
                new_state = spin_models.simulate_OAT(new_params, num_qubits)
                assert np.allclose(state, transformation.transform(new_state), atol=1e-6)
