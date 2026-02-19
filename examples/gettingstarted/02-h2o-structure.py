""".. _h2o_structure_opt:

Structure optimization: :mol:`H_2O`
===================================

Let's calculate the structure of the :mol:`H_2O` molecule.
Part of this tutorial is exercise-based, so refer back to what you learnt
from :ref:`introductionexample` and :ref:`atomscalculators`.
Suggested solutions to the exercises are found after each exercise,
but try solving them youself first!
"""
# %%
# .. admonition:: Exercise
#
#    Create an :class:`~ase.Atoms` object representing an :mol:`H_2O`
#    molecule by providing chemical symbols and a guess for the positions.
#    Visualize it, making sure the molecule is V shaped.
#
# Solution:

import matplotlib.pyplot as plt

from ase import Atoms
from ase.visualize.plot import plot_atoms

atoms = Atoms('HOH', positions=[[0, 0, -1], [0, 1, 0], [0, 0, 1]])
atoms.center(vacuum=3.0)

fig, ax = plt.subplots()
plot_atoms(atoms, ax, rotation='10x,60y,0z')
ax.set_axis_off()

# %%
# .. admonition:: Exercise
#
#    Run a self-consistent calculation of the approximate :mol:`H_2O` molecule
#    using GPAW.
#
# Solution:

from gpaw import GPAW

calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt')
atoms.calc = calc

# %%
# Optimizers
# ----------
# We will next want to optimize the geometry.
# ASE provides :mod:`several optimization algorithms <ase.optimize>`
# that can run on top of :class:`~ase.Atoms` equipped with a calculator:
#
# .. admonition:: Exercise
#
#    Run a structure optimization, thus calculating the equilibrium
#    geometry of :mol:`H_2O`.
#
# Solution:

from ase.optimize import BFGS

opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')
opt.run(fmax=0.05)

# %%
# The ``trajectory`` keyword above ensures that the trajectory of intermediate
# geometries is written to :file:`opt.traj`.
#
# .. admonition:: Exercise
#
#   Visualize the output trajectory and play it as an animation.
#   Use the mouse to drag a box around and select the three atoms —
#   this will display the angles between them.
#   What is H–O–H angle of :mol:`H_2O`?
#
# Solution:
#
# .. code-block:: python
#
#    from ase.io import read
#    from ase.visualize import view
#
#    atoms = read('opt.traj', ':')
#    view(atoms)

# %%
# Note that the above will open in a separate graphical window.
# As always in ASE, we can do things programmatically, too,
# if we know the right incantations:

from ase.io import read
atoms = read('opt.traj', ':')
print(atoms[-1].get_angle(0, 1, 2))
print(atoms[-1].get_angle(2, 0, 1))
print(atoms[-1].get_angle(1, 2, 0))

# %%
# The documentation on the :class:`~ase.Atoms` object provides
# a long list of methods.
#
# G2 molecule dataset
# -------------------
#
# ASE knows many common molecules, so we did not really need to type in
# all the molecular coordinates ourselves.  As luck would have it, the
# :func:`ase.build.molecule` function does exactly what we need:

from ase.build import molecule

atoms = molecule('H2O', vacuum=3.0)
fig, ax = plt.subplots()
plot_atoms(atoms, ax, rotation='10x,60y,0z')
ax.set_axis_off()

# %%
# This function returns a molecule from the G2 test set, which is nice
# if we remember the exact name of that molecule, in this case :mol:`'H_2O'`.
# In case we don't have all the molecule names memorized, we can work
# with the G2 test set using the more general :mod:`ase.collections.g2`
# module:

from ase.collections import g2

print(g2.names)  # These are the molecule names
atoms = g2['CH3CH2OH']

# %%
# To visualize the selected molecule as well as all 162 systems, run
#
# .. code-block:: python
#
#    view(atoms)
#    view(g2)

# %%
# Use another calculator
# ----------------------
#
# We could equally well substitute
# another calculator, often accessed through imports like ``from
# ase.calculators.emt import EMT`` or ``from ase.calculators.aims import
# Aims``.  For a list, see :mod:`ase.calculators` or run::
#
#    $ ase info --calculators
