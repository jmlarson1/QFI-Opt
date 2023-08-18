import dataclasses
import functools
from typing import Callable, Optional, Sequence

import jax.numpy as np
import numpy

from qfi_opt import spin_models


@dataclasses.dataclass(kw_only=True)
class Transformation:
    """An object representing a sequence of transformations to apply to a quantum state."""

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


# Type signature for the parameters of a sensing protocol.
Params = tuple[float, float, float, float, float]

# A symmetry `S` maps a set of parameters `p_old` to a set of parameters `p_new` together with a transformation `T`, i.e., `S: p_old --> (p_new, T)`.
# The symmetry is such that the following two procedures yield the same state:
# - Running a sensing protocol with parameters `p_old`.
# - Running the same sensing protocol with parameters `p_new`, and applying the transformation `T` to the resulting state.
# In other words, if `rho(p)` is the state returned by a sensing protocol with parameters `p`, then `rho(p_old) = T(rho(p_new))`.
Symmetry = Callable[[float, float, float, float, float], tuple[Params, Transformation]]


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


def get_symmetries_common() -> Sequence[Symmetry]:
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


def get_symmetries_U1_Z2() -> Sequence[Symmetry]:
    """Generate a list of symmetries for protocol with an axial U(1) and transverse Z_2 symmetry."""

    def shift_axes(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        axis_shift = numpy.random.random()
        final_rz = 2 * np.pi * axis_shift
        return (t_1, a_1 - axis_shift, t_ent, t_2, a_2 - axis_shift), Transformation(final_rz=final_rz)

    def shift_1(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (t_1 + 1, a_1, t_ent, t_2, 2 * a_1 - a_2), Transformation(flip_xy=a_1)

    def conjugate(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (-t_1, -a_1, -t_ent, -t_2, -a_2), Transformation(conjugate=True)

    return [shift_axes, shift_1, conjugate]


def get_symmetries_Z2_Z2() -> Sequence[Symmetry]:
    """Generate a list of symmetries for protocol with an axial Z_2 and transverse Z_2 symmetry."""

    def rot_z(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = -np.pi / 2
        return (t_1, a_1 + 0.25, -t_ent, t_2, a_2 + 0.25), Transformation(final_rz=final_rz)

    def pi_x(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = 4 * np.pi * a_2
        return (t_1 + 1, -a_1, -t_ent, t_2 + 1, -a_2), Transformation(final_rz=final_rz)

    def conjugate(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        return (-t_1, -a_1, t_ent, -t_2, -a_2), Transformation(conjugate=True)

    return [rot_z, pi_x, conjugate]


def get_symmetries_OAT(even_qubit_number: bool) -> Sequence[Symmetry]:
    """Generate a list of symmetries unique to the OAT protocol."""

    def shift_ent(t_1: float, a_1: float, t_ent: float, t_2: float, a_2: float) -> tuple[Params, Transformation]:
        final_rz = np.pi * even_qubit_number
        return (t_1, a_1, t_ent + 1, t_2, a_2 + 0.5 * even_qubit_number), Transformation(final_rz=final_rz)

    return [shift_ent]


def run_symmetry_tests(simulate_method: Callable[[Sequence[float]], np.ndarray], symmetries: Sequence[Symmetry], atol: float = 1e-6) -> None:
    """Test that the given simulation method obeys the given symmetries."""
    params = tuple(numpy.random.random(5))
    state = simulate_method(params)
    for symmetry in symmetries:
        new_params, transformation = symmetry(*params)
        new_state = simulate_method(new_params)
        assert np.allclose(state, transformation.transform(new_state), atol=atol)


def test_symmetries() -> None:
    """Test the symmetry transformations that we use to cut down the parameter domain of sensing protocols."""
    for _ in range(5):  # test several random instances
        coupling_op = get_random_hamiltonian(4)
        coupling_exponent = numpy.random.random() * 3

        for num_qubits in [2, 3]:  # test both even and odd qubit numbers
            # test common symmetries
            run_symmetry_tests(
                lambda params: spin_models.simulate_spin_chain(params, num_qubits, coupling_op, coupling_exponent),
                get_symmetries_common(),
            )

            # test U(1) x Z_2 symmetries
            for simulate_method in [
                lambda params: spin_models.simulate_ising_chain(params, num_qubits, coupling_exponent),
                lambda params: spin_models.simulate_XX_chain(params, num_qubits, coupling_exponent),
            ]:
                run_symmetry_tests(simulate_method, get_symmetries_U1_Z2())

            # test Z_2 x Z_2 symmetries
            run_symmetry_tests(
                lambda params: spin_models.simulate_local_TAT_chain(params, num_qubits, coupling_exponent),
                get_symmetries_Z2_Z2(),
            )

            # test OAT symmetries
            run_symmetry_tests(
                lambda params: spin_models.simulate_OAT(params, num_qubits),
                get_symmetries_OAT(num_qubits % 2 == 0),
            )
