""".. _surface:

ASE Introduction: Nitrogen on copper
====================================

This section gives a quick (and incomplete) overview of what ASE can do.

We will calculate the adsorption energy of a nitrogen
molecule on a copper surface.
This is done by calculating the total
energy for the isolated slab and for the isolated molecule. The
adsorbate is then added to the slab and relaxed, and the total energy
for this composite system is calculated. The adsorption energy is
obtained as the sum of the isolated energies minus the energy of the
composite system.
"""

# %%
# Here is a picture of the system after the relaxation:
#
# .. image:: ../../gettingstarted/surface.png
#
# Please have a look at the following script
# :download:`../../gettingstarted/N2Cu.py`:
#
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(4, 4, 2), vacuum=10.0)

slab.calc = EMT()
e_slab = slab.get_potential_energy()

molecule = Atoms('2N', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, d)])
molecule.calc = EMT()
e_N2 = molecule.get_potential_energy()

add_adsorbate(slab, molecule, h, 'ontop')
constraint = FixAtoms(mask=[a.symbol != 'N' for a in slab])
slab.set_constraint(constraint)
dyn = QuasiNewton(slab, trajectory='N2Cu.traj')
dyn.run(fmax=0.05)

print('Adsorption energy:', e_slab + e_N2 - slab.get_potential_energy())

# %%
#
# Assuming you have ASE setup correctly (:ref:`download_and_install`)
# run the script::
#
#  python N2Cu.py
#
# Please read below what the script does.
#
# Atoms
# -----
#
# The :class:`~ase.Atoms` object is a collection of atoms.  Here
# is how to define a N2 molecule by directly specifying the position of
# two nitrogen atoms::


d = 1.10
molecule = Atoms('2N', positions=[(0.0, 0.0, 0.0), (0.0, 0.0, d)])
