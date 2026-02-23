""".. _vibrational_modes:

Vibrational Modes of a Molecule
===============================

Let's calculate the vibrational modes of :mol:`H_2O`.

Consider the molecule at its equilibrium positions.  If we displace the
atoms slightly, the energy :math:`E` will increase, and restoring
forces will make the atoms oscillate in some pattern around the equilibrium.
"""

# %%
# We can Taylor expand the energy with respect to the 9 coordinates
# (generally :math:`3N` coordinates for a molecule with
# :math:`N` atoms), :math:`u_i`:
#
# .. math::
#
#   E = E_0 + \frac{1}{2}\sum_{i}^{3N} \sum_{j}^{3N}
#   \frac{\partial^2 E}{\partial u_{i}\partial u_{j}}\bigg\rvert_0
#   (u_i - u_{i0}) (u_j - u_{j0}) + \cdots
#
# Since we are expanding around the equilibrium positions, the energy
# should be stationary and we can omit linear contributions.
#
# The matrix of all the second derivatives is called the Hessian,
# :math:`\mathbf H`, and it expresses a linear system of differential equations
#
# .. math::
#
#  \mathbf{Hu}_k = \omega_k^2\mathbf{Mu}_k
#
# for the vibrational eigenmodes :math:`u_k` and their frequencies
# :math:`\omega_k` that will characterise the collective movement of the atoms.
# In short, we need the eigenvalues and eigenvectors of the Hessian.
#
# The elements of the Hessian can be approximated as
#
# .. math::
#
#  H_{ij} = \frac{\partial^2 E}{\partial u_{i}\partial u_{j}}\bigg\rvert_0
#    = -\frac{\partial F_{j}}{\partial u_{i}},
#
# where :math:`F_j` are the forces.  Hence we calculate the derivative
# of the forces using finite differences.
# We need to displace each atom back and forth along each Cartesian direction,
# calculating forces at each configuration to establish
# :math:`H_{ij} \approx \Delta F_{j} / \Delta u_{i}`,
# then get eigenvalues and vectors of that.
#
#
# ASE provides the :class:`~ase.vibrations.Vibrations` class for this
# purpose.  Note how the linked documentation contains an example for
# the :mol:`N_2` molecule, which means we almost don't have to do any
# work ourselves.  We just scavenge the online ASE
# documentation like we always do, then hack as necessary until the thing runs.
#
#
# .. admonition:: Exercise
#
#   Calculate the vibrational frequencies of :mol:`H_2O` using GPAW in
#   LCAO mode, saving the modes to trajectory files.  What are the
#   frequencies, and what do the eigenmodes look like?

# %%
# Solution
# ---------
# First, we import the necessary examples and build a water structure.

from math import cos, pi, sin

from gpaw import GPAW

from ase import Atoms
from ase.build import molecule
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

# Water molecule:
h2o = molecule('H2O', vacuum=3.5)
d = 0.9575
t = pi / 180 * 104.51

h2o = Atoms(
    'H2O', positions=[(0, 0, 0), (d, 0, 0), (d * cos(t), d * sin(t), 0)]
)

h2o.center(vacuum=3.5)

# %%
# Next, we attach a calculator (here: GPAW) and compute the vibrational
# modes of the water molecule.

h2o.calc = GPAW(txt='h2o.txt', mode='lcao', basis='dzp', symmetry='off')

QuasiNewton(h2o).run(fmax=0.05)


"""Calculate the vibrational modes of a H2O molecule."""

# Create vibration calculator
vib = Vibrations(h2o)
vib.run()
vib.summary(method='frederiksen')

# Make trajectory files to visualize normal modes:
for mode in range(9):
    vib.write_mode(mode)

# %%
# Since there are nine coordinates, we get nine eigenvalues and
# corresponding modes.  However the three translational and three
# rotational degrees of freedom will contribute six "modes" that do not
# correspond to true vibrations.  In principle there are no restoring
# forces if we translate or rotate the molecule, but these will
# nevertheless have different energies (often imaginary) because of
# various artifacts of the simulation such as the grid used to represent
# the density, or effects of the simulation box size.
#
# Let's visualize the last vibrational mode of water, which corresponds
# to assymetric stretching of the H-O bonds.

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from ase.io import read
from ase.visualize.plot import plot_atoms

# sphinx_gallery_thumbnail_number = -1

configs = read('vib.8.traj', ':')
fig, ax = plt.subplots()
plot_atoms(h2o, ax, rotation=('0x,0y,0z'))
ax.set_axis_off()


def animate(i):
    # Remove the previous atomic plot
    [p.remove() for p in ax.patches]
    plot_atoms(configs[i], ax, rotation=('0x,0y,0z'), show_unit_cell=1)
    ax.set_xlim(-4, 7)
    ax.set_ylim(-4, 7)
    ax.set_axis_off()
    return (ax,)


ani = animation.FuncAnimation(
    fig, animate, repeat=True, frames=len(configs) - 1, interval=200
)


# %%
# This solution is based on an example from the GPAW web page,
# where also other comments to this exercise can be found:
#
# `GPAW website <https://gpaw.readthedocs.io/tutorialsexercises/vibrational/vibrations/vibrations.html>`__
