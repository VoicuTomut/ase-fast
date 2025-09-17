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

import numpy as np

from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import QuasiNewton
from ase.optimize.fire import FIRE

# %%
# First, we create the initial state: An N$_{2}$ molecule on a Cu(111) slab

# Set up a (4 x 4) two layer Cu (111) slab
slab = fcc111('Cu', size=(4, 4, 2))
slab.set_pbc((1, 1, 0))

# Add the N2 Molecule,  oriented at 60 degrees:
d = 1.10  # N2 bond length
N2mol = Atoms(
    'N2', positions=[[0.0, 0.0, 0.0], [0.5 * 3**0.5 * d, 0.5 * d, 0.0]]
)

# %%
# Then we put the adsorbate onto the surface
#

add_adsorbate(slab, N2mol, height=1.0, position='fcc')

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
# Next, we make the NEB object, and interpolate to guess
# the intermediate steps


band = NEB(configs)
band.interpolate()

# %%
# Then we relax the configurations in the NEB object


relax = FIRE(band)
relax.run()

# %%
# Finally, we compare intermediate steps to the initial energy

e0 = initial.get_potential_energy()
for i, config in enumerate(configs):
    d = np.linalg.norm(config[-2].position - config[-1].position)
    delta_energy = config.get_potential_energy() - e0
    print(f'{i:>3}\td = {d:>4.3f}AA \tdelta energy = {delta_energy:>5.3f}eV')

# %% After the calculation is complete, the energy difference
# with respect to the initial state is given for each image,
# as well as the distance between the N atoms.
