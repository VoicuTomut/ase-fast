""".. _constraints diffusion tutorial:

============================================================
Constrained Calculations - Surface diffusion energy barriers
============================================================

In this tutorial, we will calculate the energy barrier that was found
using the :mod:`NEB <ase.mep.neb>` method in the :ref:`diffusion tutorial`
tutorial.  Here, we use a simple :class:`~ase.constraints.FixedPlane`
constraint that forces the Au atom to relax in the *yz*-plane only:
"""

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import QuasiNewton

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# %%
# We can visualize the structure with ase visualize:
import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_atoms(slab, ax1)
plot_atoms(slab, ax2, rotation='-90x')
ax1.set_title('top view')
ax2.set_title('side view')
ax1.set_axis_off()
ax2.set_axis_off()

# %%
# Alternatively, you can use also use view directly:

#   $ from ase.visualize import view
#   $ view(slab)

# %%
# We can now continue fixing the atoms in the slab:

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
# print(mask)
fixlayers = FixAtoms(mask=mask)

# Constrain the last atom (Au atom) to move only in the yz-plane:
plane = FixedPlane(-1, (1, 0, 0))

slab.set_constraint([fixlayers, plane])

# %%
# Now we can perform the calculation optimizing the displacement
# of the gold atom along the x-axis.
# We do structure optimization here using the EMT potential:

# Use EMT potential:
slab.calc = EMT()

for i in range(5):
    qn = QuasiNewton(slab, trajectory=f'mep{i}.traj')
    qn.run(fmax=0.05)
    # Move gold atom along x-axis:
    slab[-1].x += slab.get_cell()[0, 0] / 8

# %%
# The result can be analysed with the command :command:`ase gui mep?.traj -n
# -1` (choose :menuselection:`Tools --> NEB`).  The barrier is found to
# be 0.35 eV - exactly as in the :ref:`NEB <diffusion tutorial>`
# tutorial.
#
# Let's visualize the saved trajectory.
# Here is code to visualize
# a side-view of the path (unit cell repeated twice):

import matplotlib.animation as animation

from ase.io import read

configs = [read(f'mep{i}.traj', '-1') for i in range(5)]

fig, axs = plt.subplots(
    nrows=2,
    gridspec_kw={'height_ratios': [0.5, 1.0]},
)

# get the potential energies of the structures
energies = [config.get_potential_energy() for config in configs]
# set last energy value to 0 for easier comparison
energies = [energy - energies[-1] for energy in energies]
# for easier visualization, let's repeat the structures
configs = [config.repeat((2, 1, 1)) for config in configs]


axs[0].plot(range(1, len(energies) + 1), energies)
scat = axs[0].scatter(range(1, len(energies) + 1), energies)
axs[0].set_ylabel(r'$E(i) - E_{\mathrm{final}}$ (meV)')
axs[0].set_xlabel('Image number')

# Plot the atomic structure, focussing on the two Nitrogen atoms
plot_atoms(configs[0], axs[1], rotation=('-90x,0y,0z'), show_unit_cell=1)
axs[1].set_axis_off()
axs[1].set_ylim(0.0, 15)
axs[1].set_xlim(0.0, 15)
axs[1].set_yticks([0.0, 0.15, 0.3])


def animate(i):
    scat.set_offsets((range(1, len(energies) + 1)[i], energies[i]))

    # Remove the previous atomic plot
    [p.remove() for p in axs[1].patches]
    plot_atoms(configs[i], axs[1], rotation=('-90x,0y,0z'), show_unit_cell=1)
    axs[1].set_axis_off()
    axs[1].set_xlim(0.0, 15)
    axs[1].set_ylim(0.0, 15)
    axs[1].set_yticks([0.0, 0.15, 0.3])
    return (scat,)


ani = animation.FuncAnimation(
    fig, animate, repeat=True, frames=len(configs), interval=500
)

# %%
# .. seealso::
#
#   * :mod:`ase.mep.neb`
#   * :mod:`ase.constraints`
#   * :ref:`diffusion tutorial`
#   * :func:`~ase.build.fcc100`
