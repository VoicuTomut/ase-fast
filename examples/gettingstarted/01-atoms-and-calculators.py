""".. _atomscalculators:

Atoms and calculators
=====================

ASE allows atomistic calculations to be scripted with different
computational codes. In this introductory exercise, we go through the
basic concepts and workflow of ASE and will eventually
calculate the binding curve of :mol:`N_2`.


Python
------

In ASE, calculations are performed by writing and running Python
scripts.  A very short primer on Python can be found in the
:ref:`ASE examples <pythonintroduction>`.
If you are new to Python it would be wise to look through
this to understand the basic syntax, datatypes, and
things like imports.  Or you can just wing it --- we won't judge.


Atoms
-----

Let's set up a molecule and run a calculation.
We can create
simple molecules by manually typing the chemical symbols and a
guess for the atomic positions in Ångström.  For example
:mol:`N_2`:

"""

from ase import Atoms

atoms = Atoms('N2', positions=[[0, 0, -1], [0, 0, 1]])

# %%
#
# Just in case we made a mistake, we should visualize our molecule
# using the :mod:`ASE GUI <ase.gui>`:
#

# %%
#
# .. code-block:: python
#
#    from ase.visualize import view
#    view(atoms)
#
# Equivalently we can save the atoms in some format, often ASE's own
# :mod:`~ase.io.trajectory` format:
#

from ase.io import write

write('myatoms.traj', atoms)

# %%
# Then run the GUI from a terminal::
#
#  $ ase gui myatoms.traj
#
# ASE supports quite a few different formats.   For the full list, run::
#
#  $ ase info --formats
#
# Although we won't be using all the ASE commands any time soon,
# feel free to get an overview::
#
#  $ ase --help
#
#
# Calculators
# -----------
#
# Next, let us perform a calculation.  ASE uses
# :mod:`~ase.calculators` to perform calculations. Calculators are
# abstract interfaces to different backends which do the actual computation.
# Normally, calculators work by calling an external electronic structure
# code or force field code.  To run a calculation, we must first create a
# calculator and then attach it to the :class:`~ase.Atoms` object.
# For demonstration purposes, we use the :class:`~ase.calculators.emt.EMT`
# calculator which is implemented in ASE.
# However, there are many other internal and external
# calculators to choose from (see :mod:`~ase.calculators`).
#

from ase.calculators.emt import EMT

calc = EMT()
atoms.calc = calc


# %%
# Once the :class:`~ase.Atoms` object have a calculator with appropriate
# parameters, we can do things like calculating energies and forces:
#

e = atoms.get_potential_energy()
print('Energy', e)
f = atoms.get_forces()
print('Forces', f)

# %%
# This will give us the energy in eV and the forces in eV/Å
# (see :mod:`~ase.units` for the standard units ASE uses).
#
# Depending on the calculator, other properties are also available to calculate.
# For this check the documentation of the respective calculator
# or print the implemented properties the following way:

print(EMT.implemented_properties)


# %%
# Binding curve of :mol:`N_2`
# ---------------------------
#
# The strong point of ASE is that things are scriptable.
# ``atoms.positions`` is a numpy array containing the atomic positions:
#

print(atoms.positions)

# %%
# We can move the nitrogen atoms by adding or assigning other values into some
# of the array elements.  ASE understands that the state of the atoms object has
# changed and therefore we can trigger a new calculation by calling
# :meth:`~ase.Atoms.get_potential_energy` or :meth:`~ase.Atoms.get_forces`
# again, without reattatching a calculator.
#

atoms.positions[0, 2] += 0.1  # z-coordinate change of atom 0
e = atoms.get_potential_energy()
print('Energy', e)
f = atoms.get_forces()
print('Forces', f)

# %%
# This way we can implement any series of calculations by changing the atoms
# object and subsequently calculating a property.  When running
# multiple calculations, we often want to write them into a file.
# We can use the standard :mod:`~ase.io.trajectory` format to write multiple
# calculations in which the atoms objects and their respective properties
# such as energy and forces are contained. Here for a single
# calculation:
#

from ase.io.trajectory import Trajectory

with Trajectory('mytrajectory.traj', 'w') as traj:
    traj.write(atoms)

# %%
# Now, we can displace one of the atoms in small steps to trace out a binding
# energy curve :math:`E(d)` around the equilibrium
# distance.  We safe each step to a single trajectory file so that we can
# evaluate the results later on separately.
#

atoms = Atoms('N2', positions=[[0, 0, -1], [0, 0, 1]])

calc = EMT()
atoms.calc = calc

step = 0.1
nsteps = int(6 / step)

with Trajectory('binding_curve.traj', 'w') as traj:
    for i in range(nsteps):
        d = 0.5 + i * step
        atoms.positions[1, 2] = atoms.positions[0, 2] + d

        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        print('distance, energy', d, e)
        print('force', f)
        traj.write(atoms)

# %%
# As before, you can use the command line interface to visualize
# the dissociation process::
#
#  $ ase gui binding_curve.traj
#
# Although the GUI will plot the energy curve for us, publication
# quality plots usually require some manual tinkering.
# ASE provides two functions to read trajectories or other files:
#
#  * :func:`ase.io.read` reads and returns the last image, or possibly a
#    list of images if the ``index`` keyword is also specified.
#
#  * :func:`ase.io.iread` reads multiple images, one at a time.
#
# Use :func:`ase.io.iread` to read the images back in, e.g.:
#

from ase.io import iread

for atoms in iread('binding_curve.traj'):
    print(atoms.get_potential_energy())

# %%
# Now, we can plot the binding curve (energy as a function of distance)
# with matplotlib and calculate the dissociation energy.
# We first collect the energies and the distances when looping
# over the trajectory.  The atoms already have the energy.  Hence, calling
# ``atoms.get_potential_energy()`` will simply retrieve the energy
# without calculating anything.
#

import matplotlib.pyplot as plt

energies = []
distances = []

for atoms in iread('binding_curve.traj'):
    energies.append(atoms.get_potential_energy())
    distances.append(atoms.positions[1, 2] - atoms.positions[0, 2])

ax = plt.gca()
ax.plot(distances, energies)
ax.set_xlabel('Distance [Å]')
ax.set_ylabel('Total energy [eV]')
plt.show()

print('Dissociation energy [eV]: ', energies[-1] - min(energies))
