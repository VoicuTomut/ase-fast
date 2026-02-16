""".. _dissociation:

Dissociation of a molecule using the NEB method
===============================================

In this tutorial we provide an illustrative
example of a nudged-elastic band (NEB) calculation.
For more information on the NEB technique, see :mod:`ase.mep.neb`.

We consider the dissociation of a nitrogen molecule on the Cu (111) surface.

The first step is to find the relaxed structures of
the initial and final states.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import QuasiNewton
from ase.optimize.fire import FIRE
from ase.visualize.plot import plot_atoms

# %%
# First, we create the initial state: An N\ :sub:`2`\  molecule on a
# Cu(111) slab

# Set up a (4 x 4) two layer Cu (111) slab
slab = fcc111('Cu', size=(4, 4, 2))
slab.set_pbc((1, 1, 0))

# Add the N2 Molecule,  oriented at 60 degrees:
d = 1.10  # N2 bond length
n2_mol = Atoms(
    'N2', positions=[[0.0, 0.0, 0.0], [0.5 * 3**0.5 * d, 0.5 * d, 0.0]]
)

# %%
# Then we put the adsorbate onto the surface
#

add_adsorbate(slab, n2_mol, height=1.0, position='fcc')

# %%
# And add a calculator for the forces and energies:

slab.calc = EMT()

# %%
# We don't want to worry about the Cu degrees of freedom,
# so fix these atoms:

mask = [atom.symbol == 'Cu' for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

# %%
# Then we relax the structure and write the trajectory into a file

relax = QuasiNewton(slab)
relax.run(fmax=0.05)
print('initial state:', slab.get_potential_energy())
write('N2.traj', slab)

# %%
# Now the final state.
# Move the second N atom to a neighboring hollow site and relax.

slab[-1].position[0] = slab[-2].position[0] + 0.25 * slab.cell[0, 0]
slab[-1].position[1] = slab[-2].position[1]

relax.run()
print('final state:  ', slab.get_potential_energy())
write('2N.traj', slab)

# %%
# Having obtained these structures we set up an NEB
# calculation with 9 images.  Using :func:`~neb.interpolate()`
# provides a guess for the path between the initial
# and final states.  We perform the relaxation of the images
# and obtain the intermediate steps.
#
# First, we read the previous configurations

initial = read('N2.traj')
final = read('2N.traj')

# %%
# Then, we make 9 images (note the use of copy) and, as before,
# we fix the Cu atoms

configs = [initial.copy() for i in range(8)] + [final]

constraint = FixAtoms(mask=[atom.symbol != 'N' for atom in initial])
for config in configs:
    config.calc = EMT()
    config.set_constraint(constraint)

# %%
# Next, we make the NEB object, and call its interpolate() method to guess
# the intermediate steps

band = NEB(configs)
band.interpolate()

# %%
# Then we relax the configurations in the NEB object. We can use same FIRE
# optimization class we might apply to optimize an Atoms object; NEB presents
# appropriate degrees of freedom and gradients for this to optimise multiple
# configurations simultaneously.

relax = FIRE(band)
relax.run()

# %%
# Finally, we compare intermediate steps to the initial energy

energy_initial = initial.get_potential_energy()
n2_distances = []
energy_differences = []
for i, config in enumerate(configs):
    n2_distance = np.linalg.norm(config[-2].position - config[-1].position)
    delta_energy = config.get_potential_energy() - energy_initial
    n2_distances.append(n2_distance)
    energy_differences.append(delta_energy)
    print(
        f'{i:>3}\td = {n2_distance:>4.3f}AA '
        f'\tdelta energy = {delta_energy:>5.3f}eV'
    )

# %%
# After the calculation is complete, the energy difference
# with respect to the initial state is given for each image,
# as well as the distance between the N atoms.
#

energy_differences = [e * 1.0e03 for e in energy_differences]  # in meV

fig, axs = plt.subplots(
    nrows=2,
    gridspec_kw={'height_ratios': [0.5, 1.0]},
)

axs[0].plot(n2_distances, energy_differences)
scat = axs[0].scatter(n2_distances[0], energy_differences[0], s=10.0**2.0)
axs[0].set_ylabel(r'$E(i) - E_{\mathrm{initial}}$ (meV)')
axs[0].set_xlabel(r'$d_{\mathrm{N}-\mathrm{N}} (\AA)$')

# Plot the atomic structure, focussing on the two Nitrogen atoms
plot_atoms(config, axs[1], rotation='0x', show_unit_cell=0)
axs[1].set_axis_off()
axs[1].set_ylim(0.0, 5.642)
axs[1].set_xlim(0.0, 14.833)


def animate(i):
    scat.set_offsets((n2_distances[i], energy_differences[i]))

    # Remove the previous atomic plot
    [p.remove() for p in axs[1].patches]
    plot_atoms(configs[i], axs[1], rotation='0x', show_unit_cell=0)
    axs[1].set_xlim(0.0, 14.833)
    axs[1].set_ylim(0.0, 5.642)
    axs[1].set_axis_off()
    return (scat,)


ani = animation.FuncAnimation(
    fig, animate, repeat=True, frames=len(configs) - 1, interval=200
)
