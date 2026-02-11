""".. _selfdiffusion_example:

NEB and Dimer method for Self-diffusion on the Al(110) surface
==============================================================

"""


# %%
# In this exercise, we will find minimum-energy paths and transition states
# using the :mod:`Nudged Elastic Band <ase.mep.neb>` method. Another method
# for finding the transition state (i.e. the highest-energy state), the Dimer
# method, will also be explored.

# %%
#
# Initialize the system
# ---------------------
#
# Al(110) surface can be generated with ASE code
#
from math import sqrt

import numpy as np

from ase import Atom, Atoms

a = 4.0614
b = a / sqrt(2)
h = b / 2

# %%
#
#   Create  :class:`~ase.Atoms` object and

initial = Atoms(
    'Al2',
    positions=[(0, 0, 0), (a / 2, b / 2, -h)],
    cell=(a, b, 2 * h),
    pbc=(1, 1, 0),
)

# %%
# Multiply the unit cell to make it larger in x,y,z
initial *= (4, 4, 2)

# %%
# You can visualize the surface using
import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

fig, ax = plt.subplots()
plot_atoms(initial, ax, rotation=('-60x, 10y,0z'))
ax.set_axis_off()

# %%
# Then, lets add the Al addatom which will be the one moving on the surface.
initial.append(Atom('Al', (a / 2, b / 2, 3 * h)))

# %%
# Center the cell in vacuum along the z axis
initial.center(vacuum=4.0, axis=2)

# %%
# Visualize the new atom in the cell
fig, ax = plt.subplots()
plot_atoms(initial, ax, rotation=('-60x, 10y,0z'))
ax.set_axis_off()


# %%
# Perform a NEB calculation
# -------------------------
#
# The adatom can jump along the rows (into the picture) or across
# the rows (to the right inthe picture).
# We are going to compute this motion to find out which of the two jump
# will have the largest energy barrier.
# To do this, you need to create an image with the atoms at
# their final position.
# First copy the initial :class:`~ase.Atoms` object
final = initial.copy()

# %%
# Then move the last atom of the :class:`~ase.Atoms` object "final"
# (the one atom we just added before) of +b along the second positional array
final.positions[-1, 1] += b

# %%
# Visualize the new atom in the cell
fig, ax = plt.subplots()
plot_atoms(final, ax, rotation=('-60x, 10y,0z'))
ax.set_axis_off()


# %%
# Let us fix the atoms that are not moving by creating a constraint
# and setting this constraint to the images.
# To do this, we create a mask of boolean array that select
# fixed atoms (the two bottom layers):

from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

mask = initial.positions[:, 2] - min(initial.positions[:, 2]) < 1.5 * h
constraint = FixAtoms(mask=mask)
print(mask)

# %%
# Set the :class:`~ase.constraints.FixAtoms` to the :class:`~ase.Atoms` objects,
# and in the same loop, set the calculator (EMT)
initial.calc = EMT()
initial.set_constraint(constraint)
final.calc = EMT()
final.set_constraint(constraint)


# %%
# Use :class:`~ase.calculators.emt` calculator and
# :class:`~ase.optimize.QuasiNewton` Algorithm to optimize
# the geometry of the initial and final states
from ase.optimize import QuasiNewton

QuasiNewton(initial).run(fmax=0.05)
QuasiNewton(final).run(fmax=0.05)


# %%
# Then, construct a list of images by copying the first image several time in
# an array and append to this list the final image
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

# %%
# Because the .copy() method does not copy the calculator, you need to set a
# new one for the created images
for image in images:
    image.calc = EMT()
    image.set_constraint(constraint)


# %%
# Create a Nudged Elastic Band (:class:`~ase.mep import NEB`) object
from ase.mep import NEB

neb = NEB(images)

# %%
# Make a starting guess for the minimum energy path by performing a linear
# interpolation from the initial to the final image
neb.interpolate()

# %%
# Perform the NEB calculation minimizing the force bellow 0.05 eV/A
from ase.optimize import MDMin

minimizer = MDMin(neb)
minimizer.run(fmax=0.05)

# %%
# Visualize the minimum energy path (MEP) in side view to see the motion
for image in images:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-90x, 90y,0z'))
    ax.set_axis_off()

# %%
# Plot the variation of potential energy
potential_energies = [image.get_potential_energy() for image in images]

fig, ax = plt.subplots()
plt.plot(
    range(len(potential_energies)),
    potential_energies - potential_energies[0],
    marker='+',
)
plt.xlabel('Image number')
plt.ylabel('Potential energy (eV)')

diff = np.max(potential_energies) - potential_energies[0]
print(f'The energy barrier is {diff:.4f} eV.')

# %%
# You can visualize the NEB path using ASE GUI after saving the
# tajectory in a file

from ase.io import write

write('neb_path.traj', images, format='traj')

# %%
# Otherwise, you can use ``ase gui neb_path.traj`` command in your terminal and
# visualize the energy curve by plotting ``i, E[i] - E[1]``.
# You now can answer those questions :

# %%
# * How is the shape of the potential (symmetric/asymmetric) and does this make
#   sense for this process (when looking at the moving adatom in the
#   simulation)?
# * What is the energy barrier?
#


# %%
#
#
# Beyond your first NEB calculation
# ----------------------------------
#
# You now can redo the same process to find the energy barrier
# to cross one row.
# The following code will produce the result (by making use of the previously
# initialized code), though we encourage you to try by yourself.


final = initial.copy()
final.positions[-1, 0] += a

# %%
# Plot the images
for image in [initial, final]:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-90x, 0y, 0z'))
    ax.set_axis_off()

# %%
#
# Construct a list of images:
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

# %%
# Make a mask of zeros and ones that select fixed atoms (the
# two bottom layers):
mask = initial.positions[:, 2] - min(initial.positions[:, 2]) < 1.5 * h
constraint = FixAtoms(mask=mask)
print(mask)

for image in images:
    # Let all images use an EMT calculator:
    image.calc = EMT()
    image.set_constraint(constraint)

# %%
# Relax the initial and final states:
QuasiNewton(initial).run(fmax=0.05)
QuasiNewton(final).run(fmax=0.05)

# %%
# Create a Nudged Elastic Band:
neb = NEB(images)

# %%
# Make a starting guess for the minimum energy path (a straight line
# from the initial to the final state):
neb.interpolate()

# %%
# Relax the NEB path:
minimizer = MDMin(neb)
minimizer.run(fmax=0.05)


# %%
# Visualize the MEP in side view to see the motion

for image in images:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-90x, 0y, 0z'))
    ax.set_axis_off()


# %%
# Plot the variation of potential energy
potential_energies = [image.get_potential_energy() for image in images]

fig, ax = plt.subplots()
plt.plot(
    range(len(potential_energies)),
    potential_energies - potential_energies[0],
    marker='+',
)
plt.xlabel('Image number')
plt.ylabel('Potential energy (eV)')

diff = np.max(potential_energies) - potential_energies[0]
print(f'The energy barrier is {diff:.4f} eV.')


# %%
#
#
# Finding the third mechanism
# ----------------------------------

# %%
# A third diffusion process can be found: Diffusion by an exchange process.
# You can read more about it in the paper listed :mod:`here <ase.mep.dimer>`.
# Find the barrier for this process, and compare the energy barrier
# with the two other ones.
# The following code will produce the result (by making use of the previously
# initialized code), though we encourage you to try by yourself.

a = 4.0614
b = a / sqrt(2)
h = b / 2
initial = Atoms(
    'Al2',
    positions=[(0, 0, 0), (a / 2, b / 2, -h)],
    cell=(a, b, 2 * h),
    pbc=(1, 1, 0),
)
initial *= (2, 2, 2)
initial.append(Atom('Al', (a / 2, b / 2, 3 * h)))
initial.center(vacuum=4.0, axis=2)


final = initial.copy()
# move adatom to row atom 14
final.positions[-1, :] = initial.positions[14]
# Move row atom 14 to the next row
final.positions[14, :] = initial.positions[-1] + [a, b, 0]

# %%
# Visualize the initial and final images
for image in [initial, final]:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-60x, 10y,0z'))
    ax.set_axis_off()

# %%
# Construct a list of images:
images = [initial]
for i in range(5):
    images.append(initial.copy())
images.append(final)

# %%
# Make a mask of zeros and ones that select fixed atoms (the
# two bottom layers):
mask = initial.positions[:, 2] - min(initial.positions[:, 2]) < 1.5 * h
constraint = FixAtoms(mask=mask)
print(mask)

# %%
# Let all images use an EMT calculator:
for image in images:
    image.calc = EMT()
    image.set_constraint(constraint)

# %%
# Relax the initial and final states:
QuasiNewton(initial).run(fmax=0.05)
QuasiNewton(final).run(fmax=0.05)

# %%
# Create a Nudged Elastic Band:
neb = NEB(images)

# %%
# Make a starting guess for the minimum energy path (a straight line
# from the initial to the final state):
neb.interpolate()

# %%
# Relax the NEB path:
minimizer = MDMin(neb)
minimizer.run(fmax=0.05)


# %%
# Visualize the MEP in side view to see the motion

for image in images:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-60x, 10y,0z'))
    ax.set_axis_off()

# %%
# Plot the variation of potential energy
potential_energies = [image.get_potential_energy() for image in images]

fig, ax = plt.subplots()
plt.plot(
    range(len(potential_energies)),
    potential_energies - potential_energies[0],
    marker='+',
)
plt.xlabel('Image number')
plt.ylabel('Potential energy (eV)')

diff = np.max(potential_energies) - potential_energies[0]
print(f'The energy barrier is {diff:.4f} eV.')


# %%
# .. hint::
#   When opening a trajectory with :program:`ase gui` with calculated energies,
#   the default plot window shows the energy versus frame number.
#   To get a better feel of the energy barrier in an NEB calculation;
#   choose  :menuselection:`Tools --> NEB`.
#   This will give a smooth curve   of the energy as a function of the NEB path
#   length, with the slope at each point estimated from the force.


# %%
#
# Performing Dimer-method calculation
# -----------------------------------
#
# In the NEB calculations above we knew the final states, so all we had
# to do was to calculate the path between the initial state
# and the final state.
# But in some cases we do not know the final state.
# Then the :mod:`Dimer method <ase.mep.dimer>` can be used to
# find the transition state.
# The result of a Dimer calculation will hence not be the complete particle
# trajectory as in the NEB output, but rather the configuration of
# the transition-state image.
#
# The following code will find the transition-state image of the
# jump along the row.

from ase.io import Trajectory
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

a = 4.0614
b = a / sqrt(2)
h = b / 2
initial = Atoms(
    'Al2',
    positions=[(0, 0, 0), (a / 2, b / 2, -h)],
    cell=(a, b, 2 * h),
    pbc=(1, 1, 0),
)
initial *= (2, 2, 2)
initial.append(Atom('Al', (a / 2, b / 2, 3 * h)))
initial.center(vacuum=4.0, axis=2)


initial_copy = initial.copy()

N = len(initial)  # number of atoms


# Make a mask of zeros and ones that select fixed atoms - the two
# bottom layers:
mask = initial.positions[:, 2] - min(initial.positions[:, 2]) < 1.5 * h
constraint = FixAtoms(mask=mask)
initial.set_constraint(constraint)

# Calculate using EMT:
initial.calc = EMT()

# Relax the initial state:
QuasiNewton(initial).run(fmax=0.05)
e0 = initial.get_potential_energy()

# To save the trajectory file
traj = Trajectory('dimer_along.traj', 'w', initial)
traj.write()

# %%
# Making dimer mask list:
d_mask = [False] * (N - 1) + [True]

# Set up the dimer:
d_control = DimerControl(
    initial_eigenmode_method='displacement',
    displacement_method='vector',
    logfile=None,
    mask=d_mask,
)
d_atoms = MinModeAtoms(initial, d_control)

# Displacement settings:
displacement_vector = np.zeros((N, 3))
# Strength of displacement along y axis = along row:
displacement_vector[-1, 1] = 0.001
# The direction of the displacement is set by the a in
# displacement_vector[-1, a], where a can be 0 for x, 1 for y and 2 for z.
d_atoms.displace(displacement_vector=displacement_vector)

# Converge to a saddle point:
dim_rlx = MinModeTranslate(d_atoms, trajectory=traj, logfile=None)
dim_rlx.run(fmax=0.001)


# %%
# Visualize the Initial state and the saddle point in side view to see
# the change

for image in [initial, initial_copy]:
    fig, ax = plt.subplots()
    plot_atoms(image, ax, rotation=('-90x, 90y,0z'))
    ax.set_axis_off()


diff = initial.get_potential_energy() - e0
print(f'The energy barrier is {diff:.4f} eV.')

# %%
# * Compare the transition-state images of the NEB and Dimer as viewed
#   in the GUI. Are they identical?
# * What is the energy barrier? How does it compare to the one found in the
#   NEB calculation?
# * Do the same as above for the jump across the row and the exchange
#   process by copying and modifying the Dimer script, while remembering
#   that you have to  give the relevant atoms a kick in a meaningful direction.
