""".. _diffusion tutorial:

=============================================================================
Surface diffusion energy barriers using the Nudged Elastic Band  (NEB) method
=============================================================================

"""

# %%
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read
from ase.mep import NEB
from ase.optimize import BFGS, QuasiNewton
from ase.parallel import world
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt

# %%
# First, set up the initial and final states:
# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:

# %%
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# %%
# Make sure the structure is correct:

# %%
fig, ax = plt.subplots()
plot_atoms(slab, ax)
ax.set_axis_off()

# %%
# Fix second and third layers:

# %%
mask = [atom.tag > 1 for atom in slab]
print(mask)
slab.set_constraint(FixAtoms(mask=mask))

# %%
# Use EMT potential:

# %%
slab.calc = EMT()

# %%
# Initial state:

# %%
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# %%
# Final state:

# %%
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)

# %%
# .. note::  Notice how the tags are used to select the constrained atoms
#
# Now, do the NEB calculation:

# %%
initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

images = [initial]
for i in range(3):
    image = initial.copy()
    image.calc = EMT()
    image.set_constraint(constraint)
    images.append(image)

images.append(final)

neb = NEB(images, method='improvedtangent')
neb.interpolate()
qn = BFGS(neb, trajectory='neb.traj')
qn.run(fmax=0.05)

# %%
# Visualize the results with::
fig, ax = plt.subplots()
plot_atoms(final, ax)
ax.set_axis_off()

# %%
# or from the command line:
# 
#    $ ase gui neb.traj@-5:
#
# and select Tools->NEB.


# %%
# You can also create a series of plots like above, that show the progression
# of the NEB relaxation, directly at the command line::
#
#   $ ase nebplot --share-x --share-y neb.traj
#
# For more customizable analysis of the output of many NEB jobs, you can use
# the :class:`ase.mep.NEBTools` class. Some examples of its use are below; the
# final example was used to make the figure you see above.

# %%
import matplotlib.pyplot as plt

from ase.io import read
from ase.mep import NEBTools

images = read('neb.traj@-5:')

nebtools = NEBTools(images)

# %%
# Get the calculated barrier and the energy change of the reaction.

# %%
Ef, dE = nebtools.get_barrier()

# %%
# Get the barrier without any interpolation between highest images.

# %%
Ef, dE = nebtools.get_barrier(fit=False)

# %%
# Get the actual maximum force at this point in the simulation.

# %%
max_force = nebtools.get_fmax()

# %%
# Create a figure like that coming from ASE-GUI.

# %%
fig = nebtools.plot_band()

# %%
# Create a figure with custom parameters.

# %%
fig = plt.figure(figsize=(5.5, 4.0))
ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
nebtools.plot_band(ax)

# %%
# .. note::
#
#   For this reaction, the reaction coordinate is very simple: The
#   *x*-coordinate of the Au atom.  In such cases, the NEB method is
#   overkill, and a simple constraint method should be used like in this
#   tutorial: :ref:`constraints diffusion tutorial`.
#
# .. seealso::
#
#   * :mod:`ase.mep.neb`
#   * :mod:`ase.constraints`
#   * :ref:`constraints diffusion tutorial`
#   * :func:`~ase.build.fcc100`
#

# %%
# Restarting NEB
# ==============
#
# Restart NEB from the trajectory file:

# %%
# read the last structures (of 5 images used in NEB)

# %%
images = read('neb.traj@-5:')

for i in range(1, len(images) - 1):
    images[i].calc = EMT()

neb = NEB(images, method='improvedtangent')
qn = BFGS(neb, trajectory='neb_restart.traj')
qn.run(fmax=0.005)

# %%
# Parallelizing over images with MPI
# ==================================
#
# Instead of having one process do the calculations for all three
# internal images in turn, it will be faster to have three processes do
# one image each. In order to be able to run python with MPI
# you need a special parallel python interpreter, for example gpaw python
# (see `GPAW parallel runs <https://gpaw.readthedocs.io/documentation/parallel_runs/parallel_runs.html>`_)
# and set ``parallel=True`` in the NEB calculation.
#
# The example below can then be run
# with ``mpiexec -p 3 gpaw python diffusion_parallel.py``:

# %%
initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])

n_images = 1  # Set to number of processes you use with mpiexec
images = [initial]
j = world.rank * n_images // world.size  # my image number
for i in range(n_images):
    image = initial.copy()
    if i == j:
        image.calc = EMT()
    image.set_constraint(constraint)
    images.append(image)
images.append(final)

neb = NEB(images, parallel=True, method='improvedtangent')
neb.interpolate()
qn = BFGS(neb, trajectory='neb.traj')
qn.run(fmax=0.05)
