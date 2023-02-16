import dataclasses
import functools
from typing import Callable, List, Optional

import numpy as np

import run_OAT

OATParams = tuple[float, float, float, float]


@dataclasses.dataclass(kw_only=True)
class Transformation:
    """An object specifying a sequence of transformations that should be applied to a quantum state."""

    flip_z: bool = False  # if 'True', apply the global spin rotation 'Rz(pi)'
    flip_xy: Optional[float] = None  # if not 'None', apply a global spin rotation of 'pi' about 'cos(phi) * X + sin(phi) Y', where 'phi = 2 * pi * flip_xy'
    conjugate: bool = False  # if 'True', complex conjugate the state


def get_symmetries(num_qubits: int) -> List[Callable[[float, float, float, float], tuple[OATParams, Transformation]]]:
    """
    Generate a list of symmetries of the OAT protocol at zero dissipation.

    Each symmetry is a map from 'old_params' --> '(new_params, transformation)', where 'transformation' indicates how a quantum state prepared with the
    'new_params' should be additionally transformed to recover an exact symmetry.
    """
    even_qubits = num_qubits % 2 == 0

    def shift_1(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1 + 1, t_OAT, t_2, -aa), Transformation(flip_xy=0)

    def shift_2(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT, t_2 + 1, aa), Transformation(flip_xy=-aa)

    def reflect_1(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (-t_1, t_OAT, t_2, aa + 0.5), Transformation(flip_z=True)

    def reflect_2(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT, -t_2, aa + 0.5), Transformation()

    def shift_OAT(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, t_OAT + 1, t_2, aa + 0.5 * even_qubits), Transformation(flip_z=even_qubits)

    def reflect_OAT(t_1: float, t_OAT: float, t_2: float, aa: float) -> tuple[OATParams, Transformation]:
        return (t_1, -t_OAT, t_2, -aa), Transformation(flip_z=True, conjugate=True)

    return [shift_1, shift_2, shift_OAT, reflect_1, reflect_2, reflect_OAT]


@functools.cache
def rot_z_mat(num_qubits: int, angle: float) -> np.ndarray:
    """
    Construct the matrix 'phase_mat' for which element-wise multiplication 'phase_mat * density_op' rotates the state 'density_op' about the Z axis by the
    given angle.
    """
    if not angle:
        return np.ones((2**num_qubits,) * 2)
    _, _, collective_Sz = run_OAT.collective_spin_ops(num_qubits)
    phase_vec = np.exp(-1j * angle * collective_Sz.diagonal())
    return phase_vec * np.conj(phase_vec[:, np.newaxis])


def test_symmetries() -> None:
    """Test the symmetry transformations that we used to cut down the domain of the parameters for the OAT protocol."""
    # test several random parameters
    for _ in range(10):
        params = np.random.random(4)
        # test both even and odd qubit numbers
        for num_qubits in [2, 3]:
            state = run_OAT.simulate_OAT(num_qubits, params)
            # test all symmetries
            for symmetry in get_symmetries(num_qubits):
                new_params, transformation = symmetry(*params)

                new_state = run_OAT.simulate_OAT(num_qubits, new_params)
                if transformation.flip_z:
                    new_state = rot_z_mat(num_qubits, np.pi) * new_state
                if transformation.flip_xy is not None:
                    phase_mat = rot_z_mat(num_qubits, 2 * np.pi * transformation.flip_xy)
                    new_state = phase_mat * (phase_mat.conj() * new_state)[::-1, ::-1]
                if transformation.conjugate:
                    new_state = new_state.conj()

                assert np.allclose(new_state, state)
