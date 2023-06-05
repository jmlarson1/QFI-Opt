import collections
import functools
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qfi_opt import spin_models

try:
    import cmocean

    def sphere_cmap(color_vals: Sequence[float]) -> np.ndarray:
        return cmocean.cm.amp(color_vals)

except ModuleNotFoundError:

    def sphere_cmap(color_vals: Sequence[float]) -> np.ndarray:
        return plt.get_cmap("inferno")(color_vals)


def get_net_spin_projections(state: np.ndarray) -> dict[float, np.ndarray]:
    """Compute the projections of a state onto manifolds of fixed net spin S."""
    num_qubits = spin_models.log2_int(state.shape[0])
    net_spin_projectors = get_net_spin_projectors(num_qubits)
    return {val: proj @ state @ proj for val, proj in net_spin_projectors.items()}


@functools.cache
def get_net_spin_projectors(num_qubits: int) -> dict[float, np.ndarray]:
    """Construct projectors onto manifolds of fixed net spin S."""
    spin_x, spin_y, spin_z = spin_models.collective_spin_ops(num_qubits)
    vals, vecs = np.linalg.eigh(spin_x @ spin_x + spin_y @ spin_y + spin_z @ spin_z)
    vals = np.round(np.sqrt(4 * vals + 1) - 1) / 2

    projectors: dict[float, np.ndarray] = collections.defaultdict(lambda: np.zeros((2**num_qubits,) * 2, dtype=complex))
    for net_spin_val, net_spin_vec in zip(vals, vecs.T):
        projectors[net_spin_val] += np.outer(net_spin_vec, net_spin_vec.conj())
    return projectors


def axis_spin_op(theta: float, phi: float, num_qubits: int) -> np.ndarray:
    """Construct the spin operator along a given axis."""
    spin_x, spin_y, spin_z = spin_models.collective_spin_ops(num_qubits)
    return np.cos(theta) * spin_z + np.sin(theta) * (np.cos(phi) * spin_x + np.sin(phi) * spin_y)


def get_polarization(state_projections: dict[float, np.ndarray], theta: float, phi: float, cutoff: float = 1e-3) -> float:
    """
    Compute the polarization of a given state in the given direction.

    The state is provided by its projections onto manifolds with fixed net spin S.
    """
    num_qubits = spin_models.log2_int(next(iter(state_projections.values())).shape[0])
    max_spin_val = num_qubits / 2
    spin_op = axis_spin_op(theta, phi, num_qubits)
    spin_op_vals, spin_op_vecs = np.linalg.eigh(spin_op)

    polarization = 0
    for net_spin_val, state_projection in state_projections.items():
        weight = net_spin_val / max_spin_val
        if weight * np.trace(state_projection) < cutoff:
            continue

        max_spin_indices = np.isclose(spin_op_vals, net_spin_val)
        for spin_op_vec in spin_op_vecs[:, max_spin_indices].T:
            polarization += weight * (spin_op_vec.conj() @ state_projection @ spin_op_vec).real

    return polarization


def husimi(  # type: ignore[no-untyped-def]
    state: np.ndarray,
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

    state_projections = get_net_spin_projections(state)
    color_vals = np.vectorize(functools.partial(get_polarization, state_projections))(theta, phi)
    vmax = np.max(abs(color_vals)) if not color_max else color_max
    norm = mpl.colors.Normalize(vmax=vmax, vmin=0)
    color_map = sphere_cmap(norm(color_vals))

    # plot sphe

    figure = plt.figure(figsize=figsize)
    if single_sphere:
        axes = [figure.add_subplot(111, projection="3d")]
    else:
        axes = [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]
    print(type(axes[0]))

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
