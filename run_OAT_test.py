import functools
from typing import Callable, List, Optional

import numpy as np

import run_OAT

OATParams = tuple[float, float, float, float]


def get_param_symmetries(num_qubits: int) -> List[Callable[[OATParams], tuple[OATParams, bool, Optional[float], bool]]]:
    """
    Generate a list of symmetries of the OAT protocol at zero dissipation.

    Each symmetry is a map from 'old_params' --> '(new_params, flip_z, flip_xy, conjugate)', where 'flip_z', 'flip_xy', and 'conjugate' are additional
    parameters to indicate how a quantum state prepared by a protocol with the 'new_params' should be additionally transformed to recover an exact symmetry:
    - 'flip_z': if 'True', apply the global spin rotation 'Rz(pi)'.
    - 'flip_xy': if 'None', do nothing.  Otherwise apply a global spin rotation of 'pi' about 'cos(phi) * X + sin(phi) Y', where 'phi = 2 * pi * flip_xy'.
    - 'conjugate': if 'True', complex conjugate the state.
    """
    even_qubits = num_qubits % 2 == 0
    return [
        lambda t_1, t_OAT, t_2, aa: ((t_1 + 1, t_OAT, t_2, -aa), False, 0, False),
        lambda t_1, t_OAT, t_2, aa: ((t_1, t_OAT, t_2 + 1, aa), False, -aa, False),
        lambda t_1, t_OAT, t_2, aa: ((-t_1, t_OAT, t_2, aa + 0.5), True, None, False),
        lambda t_1, t_OAT, t_2, aa: ((t_1, t_OAT, -t_2, aa + 0.5), False, None, False),
        lambda t_1, t_OAT, t_2, aa: ((t_1, t_OAT + 1, t_2, aa + int(even_qubits) / 2), even_qubits, None, False),
        lambda t_1, t_OAT, t_2, aa: ((-t_1, -t_OAT, -t_2, -aa), False, None, True),
    ]


@functools.cache
def rot_z_mat(num_qubits: int, angle: float) -> None:
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
        params = tuple(np.random.random(4))
        # test both even and odd qubit numbers
        for num_qubits in [2, 3]:
            state = run_OAT.simulate_OAT(num_qubits, params)

            for symmetry in get_param_symmetries(num_qubits):
                new_params, flip_z, flip_xy, conjugate = symmetry(*params)

                new_state = run_OAT.simulate_OAT(num_qubits, new_params)
                if flip_z:
                    new_state = rot_z_mat(num_qubits, np.pi) * new_state
                if flip_xy is not None:
                    phase_mat = rot_z_mat(num_qubits, flip_xy * 2 * np.pi)
                    new_state = phase_mat * (phase_mat.conj() * new_state)[::-1, ::-1]
                if conjugate:
                    new_state = new_state.conj()

                assert np.allclose(new_state, state)
