import dataclasses
import functools
from typing import Callable, List, Optional

import jax.numpy as np
import numpy

from qfi_opt import spin_models

Params = tuple[float, float, float, float, float]


@dataclasses.dataclass(kw_only=True)
class Transformation:
    """An object representing a sequence of transformations of a quantum state."""

    final_rz: float = 0
    flip_xy: Optional[float] = None
    conjugate: bool = False

    def transform(self, state: np.ndarray) -> np.ndarray:
        num_qubits = spin_models.log2_int(state.shape[0])
        new_state = state.copy()
        if self.final_rz:
            # apply a global spin rotation 'Rz(self.final_rz)'
            phase_mat = rot_z_mat(num_qubits, self.final_rz)
            new_state = phase_mat * new_state
        if self.flip_xy is not None:
            # apply a global spin rotation by an angle 'pi' about a specified axis in the X-Y plane
            phase_mat = rot_z_mat(num_qubits, 2 * np.pi * self.flip_xy)
            new_state = phase_mat * (phase_mat.conj() * new_state)[::-1, ::-1]
        if self.conjugate:
            # complex conjugate the state
            new_state = new_state.conj()
        return new_state


def get_random_hamiltonian(dim: int) -> np.ndarray:
    """Construct a random Hamiltonian on a system of with the given dimension."""
    ham = numpy.random.random((dim, dim)) + 1j * numpy.random.random((dim, dim))
    return np.array(ham + ham.conj().T) / 2


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
    return phase_vec[:, np.newaxis] * np.conj(phase_vec)


def get_symmetries_common() -> List[Callable[..., tuple[Params, Transformation]]]:
    """
    Generate a list of parameter symmetries common to all protocols.

    Assumes zero dissipation.

    Each symmetry is a map from 'old_params' --> '(new_params, transformation)', where 'transformation' indicates how a quantum state prepared with the
    'new_params' should be additionally transformed to recover an exact symmetry.  Note that the additional transformations (should) have no effect on the QFI.
    """

    def reflect_1(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (-t_1, a_1 + 0.5, t_ent, t_2, a_2), Transformation()

    def reflect_2(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (t_1, a_1, t_ent, -t_2, a_2 + 0.5), Transformation()

    def shift_2(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (t_1, a_1, t_ent, t_2 + 1, a_2), Transformation(flip_xy=a_2)

    return [reflect_1, reflect_2, shift_2]


def get_symmetries_OAT(even_qubit_number: bool) -> List[Callable[..., tuple[Params, Transformation]]]:
    """Generate a list of symmetries unique to the OAT protocol."""

    def shift_ent(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = np.pi * even_qubit_number
        return (t_1, a_1, t_ent + 1, t_2, a_2 + 0.5 * even_qubit_number), Transformation(final_rz=final_rz)

    return [shift_ent]


def get_symmetries_U1_Z2() -> List[Callable[..., tuple[Params, Transformation]]]:
    """Generate a list of symmetries for protocol with U(1) x Z_2 symmetry."""

    def eliminate_axis(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = 2 * np.pi * a_1
        return (t_1, 0, t_ent, t_2, a_2 - a_1), Transformation(final_rz=final_rz)

    def shift_1(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (t_1 + 1, a_1, t_ent, t_2, 2 * a_1 - a_2), Transformation(flip_xy=a_1)

    def conjugate(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (-t_1, -a_1, -t_ent, -t_2, -a_2), Transformation(conjugate=True)

    return [eliminate_axis, shift_1, conjugate]


def get_symmetries_Z2_Z2() -> List[Callable[..., tuple[Params, Transformation]]]:
    """Generate a list of symmetries for protocol with Z_2 x Z_2 symmetry."""

    def reflect_z(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (t_1, a_1 + 0.5, t_ent, t_2, a_2 + 0.5), Transformation(final_rz=np.pi)

    def reflect_x(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = 4 * np.pi * a_2
        return (t_1 + 1, -a_1, -t_ent, t_2 + 1, -a_2), Transformation(final_rz=final_rz)

    def conjugate(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (-t_1, -a_1, t_ent, -t_2, -a_2), Transformation(conjugate=True)

    return [reflect_z, reflect_x, conjugate]


def test_symmetries(atol: float = 1e-6) -> None:
    """Test the symmetry transformations that we used to cut down the domain of the parameters for the OAT protocol."""
    for _ in range(5):  # test several random instances
        params = list(numpy.random.random(5))
        coupling_op = get_random_hamiltonian(4)
        coupling_exponent = numpy.random.random() * 3

        for num_qubits in [2, 3]:  # test both even and odd qubit numbers
            # test common symmetries
            state = spin_models.simulate_spin_chain(params, num_qubits, coupling_op, coupling_exponent)
            for symmetry_common in get_symmetries_common():
                new_params, transformation = symmetry_common(*params)
                new_state = spin_models.simulate_spin_chain(new_params, num_qubits, coupling_op, coupling_exponent)
                assert np.allclose(state, transformation.transform(new_state), atol=atol)

            # test OAT symmetries
            state = spin_models.simulate_OAT(params, num_qubits)
            for symmetry in get_symmetries_U1_Z2() + get_symmetries_OAT(num_qubits % 2 == 0):
                new_params, transformation = symmetry(*params)
                new_state = spin_models.simulate_OAT(new_params, num_qubits)
                assert np.allclose(state, transformation.transform(new_state), atol=atol)

            # test TAT symmetries
            state = spin_models.simulate_TAT(params, num_qubits)
            for symmetry in get_symmetries_Z2_Z2():
                new_params, transformation = symmetry(*params)
                new_state = spin_models.simulate_TAT(new_params, num_qubits)
                assert np.allclose(state, transformation.transform(new_state), atol=atol)
