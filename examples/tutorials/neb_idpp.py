""".. _neb_idpp_tutorial:

======================
NEB with IDPP: |subst|
======================

.. |subst| replace:: Image Dependent Pair Potential
                     for improved interpolation of NEB initial guess

Reference: S. Smidstrup, A. Pedersen, K. Stokbro and H. Jonsson,
:doi:`Improved initial guess for minimum energy path calculations
<10.1063/1.4878664>`,
J. Chem. Phys. 140, 214106 (2014).

Use of the Nudged Elastic Band
(NEB) method for transition state search
is dependent upon generating an initial guess
for the images lying between the initial and final states. The most
simple approach is to use linear interpolation of the
atomic coordinates. However, this can be problematic as the quality
of the interpolated path can ofter be far from the real one.
The implication being
that a lot of time is spent in the NEB routine optimising the shape of
the path, before the transition state is homed-in upon.

The image dependent pair potential (IDPP) is a method that has been
developed to provide an improvement to the initial guess for the NEB path.
The IDPP method uses the bond distance between the atoms involved in
the transition state to create target structures for the images, rather
than interpolating the atomic positions.
By defining an objective function in terms
of the distances between atoms, the NEB algorithm is used with this
image dependent pair potential to create the initial guess for the
full NEB calculation.

.. note::

   The examples below utilise the EMT calculator for illustrative purposes, the
   results should not be over interpreted.

This tutorial includes example NEB calculations for two different systems.
First, it starts with a simple NEB of Ethane comparing IDPP
to the standard linear approach.
The second example is for a N atom on a Pt step edge.

Example 1: Ethane
=================

This example illustrates the use of the IDPP interpolation scheme to
generate an initial guess for rotation of a methyl group around the CC bond.


1.1 Generate Initial and Final State
------------------------------------
"""

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.mep import NEB
from ase.optimize.fire import FIRE

# Create the molecule.
initial = molecule('C2H6')
# Attach calculators
initial.calc = EMT()
# Relax the molecule
relax = FIRE(initial)
relax.run(fmax=0.05)

# %%
# Let's look at the relaxed molecule.

import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

fig, ax = plt.subplots()
plot_atoms(initial, ax, rotation=('90x,0y,90z'))
ax.set_axis_off()

# %%
# Now we can create the final state.
# Since we want to look at the rotation
# of the methyl group, we switch the position of the
# Hydrogen atoms on one methyl group.
# Then, we setup and run the NEB calculation.

final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]

# %%
# 1.2 Linear Interpolation Approach
# ---------------------------------
#
# Generate blank images.
images = [initial]

for i in range(9):
    images.append(initial.copy())

for image in images:
    image.calc = EMT()

images.append(final)

# Run linear interpolation.
neb = NEB(images)
neb.interpolate()

# Run NEB calculation.
qn = FIRE(neb, trajectory='ethane_linear.traj')
qn.run(fmax=0.05)
# You can add a logfile to the FIRE optimizer by adding
# e.g. `logfile='ethane_linear.log` to save the printed output.

# %%
# Using the standard linear interpolation approach,
# as in the following example, we can see
# that 47 iterations are required to find the transition state.


# %%
# 1.3 Image Dependent Pair Potential
# ----------------------------------
#
# However if we modify our script slightly and use the IDPP method to
# find the initial guess, we can see that the number of iterations
# required to find the transition state is reduced to 7.

# Optimise molecule.
initial = molecule('C2H6')
initial.calc = EMT()
relax = FIRE(initial, logfile='opt.log')
relax.run(fmax=0.05)

# Create final state.
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]

# Generate blank images.
images = [initial]

for i in range(9):
    images.append(initial.copy())

for image in images:
    image.calc = EMT()

images.append(final)

# Run IDPP interpolation.
neb = NEB(images)
neb.interpolate('idpp')

# Run NEB calculation.
qn = FIRE(neb, trajectory='ethane_idpp.traj')
qn.run(fmax=0.05)

# %%
# Clearly, if one was using a full DFT calculator one can
# potentially gain a significant time improvement.
#
# Example 2: N Diffusion over a Step Edge
# =======================================
#
# Often we are interested in generating an initial guess for a surface reaction.
#
# 2.1 Generate Initial and Final State
# ------------------------------------
# The first part of this example
# illustrates how we can optimise our initial and final state structures
# before using the IDPP interpolation to generate our initial guess
# for the NEB calculation:

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.mep import NEB
from ase.optimize.fire import FIRE as FIRE

# Set the number of images you want.
nimages = 5

# Some algebra to determine surface normal and the plane of the surface.
d3 = [2, 1, 1]
a1 = np.array([0, 1, 1])
d1 = np.cross(a1, d3)
a2 = np.array([0, -1, 1])
d2 = np.cross(a2, d3)

# Create the slab.
slab = FaceCenteredCubic(
    directions=[d1, d2, d3], size=(2, 1, 2), symbol=('Pt'), latticeconstant=3.9
)

# Add some vacuum to the slab.
uc = slab.get_cell()
uc[2] += [0.0, 0.0, 10.0]  # There are ten layers of vacuum.
uc = slab.set_cell(uc, scale_atoms=False)

# Some positions needed to place the atom in the correct place.
x1 = 1.379
x2 = 4.137
x3 = 2.759
y1 = 0.0
y2 = 2.238
z1 = 7.165
z2 = 6.439


# Add the adatom to the list of atoms and set constraints of surface atoms.
slab += Atoms('N', [((x2 + x1) / 2, y1, z1 + 1.5)])
FixAtoms(mask=slab.symbols == 'Pt')

# Optimise the initial state: atom below step.
initial = slab.copy()
initial.calc = EMT()
relax = FIRE(initial, logfile='opt.log')
relax.run(fmax=0.05)

# %%
# Now let's visualize this.

fig, ax = plt.subplots()
plot_atoms(initial, ax, rotation=('0x,0y,0z'))
ax.set_axis_off()

# %%
# We can now create and optimise the final state by
# moving the atom above the step.
slab[-1].position = (x3, y2 + 1.0, z2 + 3.5)
final = slab.copy()
final.calc = EMT()
relax = FIRE(final, logfile='opt.log')
relax.run(fmax=0.05)

# %%
# Now let's visualize this.

fig, ax = plt.subplots()
plot_atoms(final, ax, rotation=('0x,0y,0z'))
ax.set_axis_off()

# %%
# 2.2 Image Dependent Pair Potential
# ----------------------------------
#
# Now we are ready to setup the NEB with the IDPP interpolation.

# Create a list of images for interpolation.
images = [initial]
for i in range(nimages):
    images.append(initial.copy())

for image in images:
    image.calc = EMT()

images.append(final)

# Carry out idpp interpolation.
neb = NEB(images)
neb.interpolate('idpp')

# Run NEB calculation.
qn = FIRE(neb, trajectory='N_diffusion.traj')
qn.run(fmax=0.05)

# %%
# 2.3 Linear Interpolation Approach
# ---------------------------------
#
# To again illustrate the potential speedup, the following script
# uses the linear interpolation.
# This takes more iterations to find a transition
# state, compared to using the IDPP interpolation.
# We start from the initial and final state we generated above.

# Create a list of images for interpolation.
images = [initial]
for i in range(nimages):
    images.append(initial.copy())

for image in images:
    image.calc = EMT()

images.append(final)

# Carry out linear interpolation.
neb = NEB(images)
neb.interpolate()

# Run NEB calculation.
qn = FIRE(neb, trajectory='N_diffusion_lin.traj')
qn.run(fmax=0.05)
