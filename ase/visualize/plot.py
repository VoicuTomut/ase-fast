# fmt: off
from __future__ import annotations

from typing import TYPE_CHECKING

from ase.io.utils import PlottingVariables, make_patch_list

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ase.atoms import Atoms


class Matplotlib(PlottingVariables):
    def __init__(self, atoms, ax,
                 rotation='', radii=None,
                 colors=None, scale=1, offset=(0, 0), **parameters):
        super().__init__(
            atoms,
            rotation=rotation,
            radii=radii,
            colors=colors,
            scale=scale,
            extra_offset=offset,
            **parameters,
        )

        self.ax = ax
        self.figure = ax.figure
        self.ax.set_aspect('equal')

    def write(self):
        self.write_body()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            self.ax.add_patch(patch)


def animate(images, ax=None,
            interval=200,  # in ms; same default value as in FuncAnimation
            save_count=None,  # ignored as of 2023 with newer matplotlib
            **parameters):
    """Convert sequence of atoms objects into Matplotlib animation.

    Each image is generated using plot_atoms().  Additional parameters
    are passed to this function."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if ax is None:
        ax = plt.gca()

    fig = ax.get_figure()

    def drawimage(atoms):
        ax.clear()
        ax.axis('off')
        plot_atoms(atoms, ax=ax, **parameters)

    animation = FuncAnimation(fig, drawimage, frames=images,
                              init_func=lambda: None,
                              interval=interval)
    return animation


def plot_atoms(atoms: Atoms, ax: Axes | None = None, **params) -> Axes:
    """Plot an Atoms object in a matplotlib subplot.

    Additional axis options for the Matplotlib axis can be provided as params,
    always starting with 'ax_' to show their affiliation to the axis.
    These are equivalent to calls to ax.set().

    Axis decorations will always be removed.

    For finer control of plot appearance or composition of custom plot
    arrangements, consider using plot_atoms_raw() to plot directly to given
    Matplotlib Axes.

    Parameters
    ----------
    atoms : Atoms object
    ax : Matplotlib subplot object
    rotation : str, optional
        In degrees. In the form '10x,20y,30z'
    show_unit_cell : int, optional, default 2
        Draw the unit cell as dashed lines depending on value:
        0: Don't
        1: Do
        2: Do, making sure cell is visible
    radii : float, optional
        The radii of the atoms
    colors : list of strings, optional
        Color of the atoms, must be the same length as
        the number of atoms in the atoms object.
    scale : float, optional
        Scaling of the plotted atoms and lines.
    offset : tuple (float, float), optional
        Offset of the plotted atoms and lines.

    Axis Parameters
    ---------------
    All axis parameters for Matplotlib objects that
    can be set by
    `Matplotlib Axes Set <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html>`_ .

    """  # noqa: E501
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    ax_parameters = {
        key[3:]: value for key, value in params.items() if key[:3] == 'ax_'
    }
    parameters = {
        key: value for key, value in params.items() if key[:3] != 'ax_'
    }

    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]

    plot_atoms_raw(atoms, ax, **parameters)
    ax.set_axis_off()
    ax.set(**ax_parameters)

    return ax


def plot_atoms_raw(atoms: Atoms, ax: Axes, **parameters) -> None:
    """Plot Atoms to a matplotlib subplot without additional formatting.

    Compared to plot_atoms this is more "barebones". To reproduce the
    additional features of plot_atoms:

    - create a figure and axes if necessary using plt.subplots()

    - disable decorations by calling ax.set_axis_off()

    - set matplotlib axis properties directly by calling ax.set()

    Parameters
    ----------
    atoms : Atoms object
    ax : Matplotlib subplot object
    rotation : str, optional
        In degrees. In the form '10x,20y,30z'
    show_unit_cell : int, optional, default 2
        Draw the unit cell as dashed lines depending on value:
        0: Don't
        1: Do
        2: Do, making sure cell is visible
    radii : float, optional
        The radii of the atoms
    colors : list of strings, optional
        Color of the atoms, must be the same length as
        the number of atoms in the atoms object.
    scale : float, optional
        Scaling of the plotted atoms and lines.
    offset : tuple (float, float), optional
        Offset of the plotted atoms and lines.

    """
    Matplotlib(atoms, ax, **parameters).write()
