""".. _atomization:

Atomization energy
==================

The following script will calculate the atomization energy of a
nitrogen molecule.
"""

from ase import Atoms
from ase.calculators.emt import EMT

# %%
# First, an ``Atoms`` object containing one nitrogen is created and a
# fast EMT calculator is attached to it simply as an argument.
#

atom = Atoms('N')
atom.calc = EMT()

# %%
# The total energy for the isolated atom is then calculated
# and stored in the ``e_atom`` variable.
#

e_atom = atom.get_potential_energy()

# %%
# The ``molecule`` object is defined, holding the nitrogen molecule at
# the experimental bond length ``d=1.1`` Angstrom.
#

d = 1.1
molecule = Atoms('2N', [(0.0, 0.0, 0.0), (0.0, 0.0, d)])

# %%
# The EMT calculator is then attached to the molecule
# and the total energy is extracted into the ``e_molecule`` variable.
#

molecule.calc = EMT()
e_molecule = molecule.get_potential_energy()

# %%
# The atomization energy is the energy required to break the bond,
# meaning that it is the negative energetic difference between
# the molecule and twice the single atom's energy.
#

e_atomization = -1.0 * (e_molecule - 2 * e_atom)

# %%
# Finally we print the relevant energies:
#

print(f'Nitrogen atom energy: {e_atom:5.2f} eV')
print(f'Nitrogen molecule energy: {e_molecule:5.2f} eV')
print(f'Atomization energy: {e_atomization:5.2f} eV')
