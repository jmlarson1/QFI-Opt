import collections
import functools
import typing
from typing import Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qfi_opt import spin_models

if typing.TYPE_CHECKING:
    import jax.numpy as jnp

STATE_TYPE = Union[np.ndarray, "jnp.ndarray"]

try:
    import cmocean

    def sphere_cmap(color_vals: Sequence[float]) -> np.ndarray:
        return cmocean.cm.amp(color_vals)

except ModuleNotFoundError:

    def sphere_cmap(color_vals: Sequence[float]) -> np.ndarray:
        return plt.get_cmap("inferno")(color_vals)


def get_spin_length_projections(state: STATE_TYPE) -> dict[float, np.ndarray]:
    """Compute the projections of a state onto manifolds of fixed spin length S.

    More specifically, compute the dictionary `{S: state_S for S in spin_length_vals}`, where
    - `state_S` = `P_S state P_S`, and
    - `P_S` is a projector onto the manifold with fixed spin length S.
    """
    num_qubits = spin_models.log2_int(state.shape[0])
    spin_length_projectors = get_spin_length_projectors(num_qubits)
    return {val: proj @ state @ proj for val, proj in spin_length_projectors.items()}


@functools.cache
def get_spin_length_projectors(num_qubits: int) -> dict[float, np.ndarray]:
    """Construct projectors onto manifolds of fixed spin length S."""
    spin_x, spin_y, spin_z = spin_models.collective_spin_ops(num_qubits)
    vals, vecs = np.linalg.eigh(spin_x @ spin_x + spin_y @ spin_y + spin_z @ spin_z)
    vals = np.round(np.sqrt(4 * vals + 1) - 1) / 2

    projectors: dict[float, np.ndarray] = collections.defaultdict(lambda: np.zeros((2**num_qubits,) * 2, dtype=complex))
    for spin_length_val, spin_length_vec in zip(vals, vecs.T):
        projectors[spin_length_val] += np.outer(spin_length_vec, spin_length_vec.conj())
    return projectors


def axis_spin_op(theta: float, phi: float, num_qubits: int) -> np.ndarray:
    """Construct the spin operator oriented along a given axis."""
    spin_x, spin_y, spin_z = spin_models.collective_spin_ops(num_qubits)
    return np.cos(theta) * spin_z + np.sin(theta) * (np.cos(phi) * spin_x + np.sin(phi) * spin_y)


def get_polarization(state_projections: dict[float, np.ndarray], theta: float, phi: float, cutoff: float = 1e-3) -> float:
    """
    Compute the polarization of a given state in the given direction.
    The polarization is defined by the average over all Husimi probability distribution functions (averaged over manifolds with fixed spin length S).

    The input state is a dictionary of the full state's projections onto manifolds of fixed spin length S.
    """
    num_qubits = spin_models.log2_int(next(iter(state_projections.values())).shape[0])
    spin_op = axis_spin_op(theta, phi, num_qubits)
    spin_op_vals, spin_op_vecs = np.linalg.eigh(spin_op)

    polarization = 0
    for spin_length_val, state_projection in state_projections.items():
        weight = 2 * spin_length_val + 1  # normalization factor for Husimi probability distribution
        if weight * np.trace(state_projection) < cutoff:
            continue

        max_spin_indices = np.isclose(spin_op_vals, spin_length_val)
        for spin_op_vec in spin_op_vecs[:, max_spin_indices].T:
            polarization += weight * (spin_op_vec.conj() @ state_projection @ spin_op_vec).real

    return polarization


def husimi(  # type: ignore[no-untyped-def]
    state: STATE_TYPE,
    grid_size: int = 101,
    single_sphere: bool = True,
    figsize: Optional[tuple[int, int]] = None,
    rasterized: bool = True,
    view_angles: tuple[float, float] = (0, 0),
    shade: bool = False,
    color_max: Optional[float] = None,
):
    if figsize is None:
        figsize = plt.figaspect(1 if single_sphere else 0.5)

    # initialize grid and color map

    theta, phi = np.meshgrid(np.linspace(0, np.pi, grid_size), np.linspace(0, 2 * np.pi, grid_size))
    z_vals = np.cos(theta)
    x_vals = np.sin(theta) * np.cos(phi)
    y_vals = np.sin(theta) * np.sin(phi)

    state_projections = get_spin_length_projections(state)
    color_vals = np.vectorize(functools.partial(get_polarization, state_projections))(theta, phi)
    vmax = np.max(abs(color_vals)) if not color_max else color_max
    norm = mpl.colors.Normalize(vmax=vmax, vmin=0)
    color_map = sphere_cmap(norm(color_vals))

    # plot sphere

    figure = plt.figure(figsize=figsize)
    if single_sphere:
        axes = [figure.add_subplot(111, projection="3d")]
    else:
        axes = [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]

    for axis, side in zip(axes, [+1, -1]):
        axis.plot_surface(side * x_vals, side * y_vals, z_vals, rstride=1, cstride=1, facecolors=color_map, rasterized=rasterized, shade=shade)

    # clean up figure

    elev, azim = view_angles
    ax_lims = np.array([-1, 1]) * 0.7
    for axis in axes:
        axis.set_xlim(ax_lims)
        axis.set_ylim(ax_lims)
        axis.set_zlim(ax_lims * 0.8)
        axis.view_init(elev=elev, azim=azim)
        axis.set_axis_off()

    left = -0.01
    right = 1
    bottom = -0.03
    top = 1
    rect = [left, bottom, right, top]
    figure.tight_layout(pad=0, w_pad=0, h_pad=0, rect=rect)
    return figure, axes


def histogram(  # type: ignore[no-untyped-def]
    state: STATE_TYPE,
    figsize: tuple[int, int] = (4, 3),
):
    """Plot the distribution function over collective-Sz measurement outcomes."""
    num_spins = spin_models.log2_int(state.shape[0])
    state_tensor = np.reshape(state, (2,) * num_spins * 2)

    # collect the probabilities for different numbers of |1> states

    probabilities = np.zeros(num_spins + 1)
    for bitstring in np.ndindex((2,) * num_spins):
        num_spins_up = num_spins - sum(bitstring)  # number of zeros in the bitstring
        tensor_index = bitstring + bitstring
        probabilities[num_spins_up] += state_tensor[tensor_index].real

    # plot a histogram and return

    measurement_outcomes = np.arange(num_spins + 1) - num_spins / 2

    figure, axis = plt.subplots(figsize=figsize)
    axis.bar(measurement_outcomes, probabilities, width=0.9)
    axis.set_ylabel("Probability")
    axis.set_xlabel("Axial spin projection")
    figure.tight_layout()
    return figure, axis
