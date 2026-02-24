""".. _md_tutorial:

==================
Molecular dynamics
==================

.. note::

  These examples *can* be run without ``asap3`` installed. In that case,
  ASE’s Python implementation of the EMT calculator can be used instead, but it
  is much slower.

Goal
====

In this tutorial, we will learn how to perform basic molecular dynamics (MD)
simulations using ASE.

The key objectives are:

- Understand how to set up a crystal structure (Cu atoms on an FCC lattice).
- Initialize velocities from Maxwell–Boltzmann distribution corresponding to a
  chosen temperature.
- Integrate Newton’s equations of motion using Velocity-Verlet algorithm and we
  monitor the temperature using Langevin thermostat.
- Monitor and analyze thermodynamic quantities (potential energy, kinetic
  energy, total energy, temperature).
- Save trajectories and visualize atomic motion with ASE’s GUI.
- Explore MD in different scenarios:
  - Constant energy MD (NVE ensemble)
  - Constant temperature MD (NVT ensemble)
  - Isolated nanoparticle simulations

By the end of this tutorial, you should be able to set up your own MD
simulations, monitor energy conservation, and visualize system evolution.

Part 1: Basic Molecular Dynamics Simulation
===========================================

We start by creating a copper crystal, assigning random velocities
corresponding to Maxwell Boltzmann Distribution at 300 K, and running dynamics
in the NVE ensemble (constant energy).

"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# choose one of the following implementations of EMT:
# included in ase
# from ase.calculators.emt import EMT
# faster performance
from asap3 import EMT

from ase import units
from ase.cluster.cubic import FaceCenteredCubic as ClusterFCC
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic as LatticeFCC
from ase.md.langevin import Langevin  # for later NPT simulations
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.optimize import QuasiNewton
from ase.visualize.plot import plot_atoms

# Set up initial positions of Cu atoms on Fcc crystal lattice
size = 10
atoms = LatticeFCC(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol='Cu',
    size=(size, size, size),
    pbc=True,
)

# %%
# Before setting up the MD simulation, we take a look at the initial structure:

# %%
fig, ax = plt.subplots(figsize=(5, 5))
plot_atoms(atoms, ax, rotation=('45x,45y,0z'), show_unit_cell=2, radii=0.75)
ax.set_axis_off()
plt.tight_layout()
plt.show()

# %%
# Now let's run the MD simulation and monitor the kinetic and potential energy
# of the whole system:

# Describe the interatomic interactions with the Effective Medium Theory (EMT)
atoms.calc = EMT()

# Set the initial velocities corresponding to T=300K from Maxwell Boltzmann
# Distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We use Velocity Verlet algorithm to integrate the Newton's equations.
timestep_fs = 5
dyn = VelocityVerlet(atoms, timestep_fs * units.fs)  # 5 fs time step.


def printenergy(a):
    """
    Function to print the thermodynamical properties i.e potential energy,
    kinetic energy and total energy
    """
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = a.get_temperature()
    print(
        f'Energy per atom: Epot ={epot:6.3f}eV  Ekin = {ekin:.3f}eV '
        f'(T={temp:.3f}K) Etot = {epot + ekin:.3f}eV'
    )


# Now run the dynamics
print('running a NVE simulation of fcc Cu')
printenergy(atoms)
# init lists to for energy vs time data
time_ps, epot, ekin = [], [], []
mdind = 0
steps_per_block = 10
for i in range(20):
    dyn.run(steps_per_block)
    mdind += steps_per_block
    printenergy(atoms)
    # save the energies of the current MD step
    time_ps.append(mdind * timestep_fs / 1000.0)
    epot.append(atoms.get_potential_energy())
    ekin.append(atoms.get_kinetic_energy())

etot = np.array(epot) + np.array(ekin)

# Plot energies vs time
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_ps, epot, label='Potential energy')
ax.plot(time_ps, ekin, label='Kinetic energy')
ax.plot(time_ps, etot, label='Total energy')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Energy (eV)')
ax.legend(loc='best')
ax.grid(True, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# Note how the total energy is conserved, but the kinetic energy quickly
# drops to half the expected value. Why?
#
# What you learned here:
#
# - How to set up a basic MD run.
# - How to monitor the energy over time.
# - That total energy is approximately conserved in NVE simulations, what is
#   the error in total energy?
#
# Exercise: Tune the time step from 5fs to 10fs and 50fs, what changes do you
# observe in total energy?


# %%
# Part 2: Constant temperature MD
# ===============================
#
# In many cases, you want to control temperature (NVT ensemble). This
# can be done using a thermostat, like -- in this tutorial -- Langevin
# thermostat.
# Compared to the previous example, we replace the line
# ``dyn = VelocityVerlet(...)`` with
#
# ::
#
#   dyn = Langevin(atoms, timestep=5 * units.fs, temperature_K=T,
#   friction=0.02)
#
# where ``T`` is the desired temperature in Kelvin. For that we also imported
# the Langevin in the beginning.
#
# The Langevin dynamics will then slowly adjust the total energy of the
# system so the temperature approaches the desired one.
#
# As a slightly less boring example, let us use this to melt a chunk of
# copper by starting the simulation without any momentum of the atoms
# (no kinetic energy), and with a desired temperature above the melting
# point. We will also save information about the atoms in a trajectory
# file called ``moldyn3.traj``.
#
# .. note::
#
#   It is recommended to use the ``asap3`` implementation of the ``EMT``
#   calculator here, because its performance benefits over the ``ase``
#   implementation.

# %%
size = 10
T = 1500  # Kelvin

# Set up a crystal
atoms = LatticeFCC(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol='Cu',
    size=(size, size, size),
    pbc=False,
)

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
timestep_fs = 5
dyn = Langevin(
    atoms, timestep=timestep_fs * units.fs, temperature_K=T, friction=0.02
)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('fccCu_NPT.traj', 'w', atoms)

# Now run the dynamics
print('running a NVT simulation of fcc Cu')
printenergy(atoms)
time_ps, temperature = [], []
mdind = 0
steps_per_block = 10
for i in range(200):
    dyn.run(steps_per_block)
    mdind += steps_per_block
    printenergy(atoms)
    # save the temperature of the current MD step
    time_ps.append(mdind * timestep_fs / 1000.0)
    temperature.append(atoms.get_temperature())

# Plot temperatures vs time
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_ps, temperature)
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Temperature (K)')
ax.grid(True, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# After running the simulation, you can study the result with the
# command
#
# ::
#
#   ase gui fccCu_NPT.traj
#
# Try plotting the kinetic energy. Like in the temperature vs time plot you
# will *not* see a well-defined melting point due to finite size effects
# (including surface melting),
# but you will probably see an almost flat region where the inside of
# the system melts. The outermost layers melt at a lower temperature.
#
# .. note::
#
#   The Langevin dynamics will by default keep the position and momentum
#   of the center of mass unperturbed. This is another improvement over
#   just setting momenta corresponding to a temperature, as we did before.
#
#
# Part 3: Isolated particle MD
# ============================
#
# When simulating isolated particles with MD, it is sometimes preferable
# to set random momenta corresponding to a specific temperature and let the
# system evolve freely. With a relatively high temperature, the is however
# a risk that the collection of atoms will drift out of the simulation box
# because the randomized momenta gave the center of mass a small but
# non-zero velocity too.
#
# Let us see what happens when we propagate a nanoparticle:

# %%
size = 4
atoms = ClusterFCC(
    'Cu',
    surfaces=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    layers=(size, size, size),
    vacuum=4,
)
# asap3 requires a non-zero cell even if pbc are not applied
atoms.cell = [40] * 3
atoms.set_pbc(False)  # isolated cluster (explicit, for clarity)

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# Quick relaxation of the cluster
qn = QuasiNewton(atoms)
qn.run(fmax=0.001, steps=10)

# Set the momenta corresponding to T=1200 K
MaxwellBoltzmannDistribution(atoms, temperature_K=1200)
Stationary(atoms)  # zero linear momentum
ZeroRotation(atoms)  # zero angular momentum

# Run MD using the Velocity Verlet algorithm and save trajectory
dyn = VelocityVerlet(atoms, 5 * units.fs, trajectory='nanoparticleCu_NVE.traj')

print('running a NVE simulation of a Cu nanoparticle')
printenergy(atoms)
steps_per_block = 10
for i in range(200):
    dyn.run(steps_per_block)
    printenergy(atoms)

# %%
# After running the simulation, use :ref:`ase-gui` to compare the resulting
# trajectory with how it looks if you comment out either the line that says
# ``Stationary(atoms)``, ``ZeroRotation(atoms)`` or both:
#
# ::
#
#   ase gui nanoparticleCu_NVE.traj
#
# Try playing the movie with a high frame rate and set frame skipping to a
# low number. Can you spot the subtle difference?
