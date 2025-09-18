""".. _md_tutorial:

==================
Molecular dynamics
==================

.. note::

  These examples *can* be run without ``asap`` installed. In that case,
  ASE’s Python implementation of the EMT calculator is used, but it is
  much slower.

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

1 Basic Molecular Dynamics Simulation
=====================================

We start by creating a copper crystal, assigning random velocities
corresponding to Maxwell Boltzmann Distribution at 300 K, and running dynamics
in the NVE ensemble (constant energy).

"""

# %%
from ase import units
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

# Use Asap for a huge performance increase if it is installed
use_asap = False

if use_asap:
    from asap3 import EMT

    size = 10
else:
    from ase.calculators.emt import EMT

    size = 3

# Set up initial positions of Cu atoms on Fcc crystal lattice
atoms = FaceCenteredCubic(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol="Cu",
    size=(size, size, size),
    pbc=True,
)

# Describe the interatomic interactions with the Effective Medium Theory (EMT)
atoms.calc = EMT()

# Set the initial velocities corresponding to T=300K from Maxwell Boltzmann
# Distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# We use Velocity Verlet algorithm to integrate the Newton's equations.
dyn = VelocityVerlet(atoms, 5 * units.fs)  # 5 fs time step.


def printenergy(a):
    """
    Function to print the thermodynamical properties i.e potential energy,
    kinetic energy and total energy
    """
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        f"Energy per atom: Epot ={epot:6.3f}eV  Ekin = {ekin:.3f}eV "
        f"(T={ekin / (1.5 * units.kB):3.0f}K) Etot = {epot + ekin:.3f}eV"
    )


# Now run the dynamics
printenergy(atoms)
for i in range(20):
    dyn.run(10)
    printenergy(atoms)

# %%
# Note how the total energy is conserved, but the kinetic energy quickly
# drops to half the expected value. Why?
#
# What you learn here:
# - How to set up a basic MD run.
# - How to monitor the energy over time.
# - That total energy is approximately conserved in NVE simulations, what is
# the error in total energy?
#
# Exercise 1: Tune the time step from 5fs to 10fs and 50fs, what you observe in
# total energy?
# Exercise 2: Change ``use_asap`` to ``True``, what differences do you see in
# the computational performance?


# %%
# 2 Constant temperature MD
# =========================
#
# In many cases, you want to control temperature (NVT ensemble).  This
# can be done using a thermostat.
# In this tutorial we will use Langevin thermostat.
# In the previous examples, replace the line ``dyn = VelocityVerlet(...)`` with
#
# ::
#
#   dyn = Langevin(atoms, timestep=5 * units.fs, temperature_K=T,
#   friction=0.002)
#
# where ``T`` is the desired temperature in Kelvin. You also need to import
# Langevin, see the class below.
#
# The Langevin dynamics will then slowly adjust the total energy of the
# system so the temperature approaches the desired one.
#
# As a slightly less boring example, let us use this to melt a chunk of
# copper by starting the simulation without any momentum of the atoms
# (no kinetic energy), and with a desired temperature above the melting
# point. We will also save information about the atoms in a trajectory
# file called ``moldyn3.traj``.

# %%
from asap3 import EMT  # Way too slow with ase.EMT !

from ase import units
from ase.io.trajectory import Trajectory
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin

size = 10

T = 1500  # Kelvin

# Set up a crystal
atoms = FaceCenteredCubic(
    directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    symbol="Cu",
    size=(size, size, size),
    pbc=False,
)

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, timestep=5 * units.fs, temperature_K=T, friction=0.002)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        f"Energy per atom: Epot ={epot:6.3f}eV  Ekin = {ekin:.3f}eV "
        f"(T={ekin / (1.5 * units.kB):4.0f}K) Etot = {epot + ekin:.3f}eV"
    )


dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory("moldyn3.traj", "w", atoms)
dyn.attach(traj.write, interval=50)

# Now run the dynamics
printenergy()
dyn.run(5000)

# %%
# After running the simulation, you can study the result with the
# command
#
# ::
#
#   ase gui moldyn3.traj
#
# Try plotting the kinetic energy. You will *not* see a well-defined
# melting point due to finite size effects (including surface melting),
# but you will probably see an almost flat region where the inside of
# the system melts.  The outermost layers melt at a lower temperature.
#
# .. note::
#
#   The Langevin dynamics will by default keep the position and momentum
#   of the center of mass unperturbed. This is another improvement over
#   just setting momenta corresponding to a temperature, as we did before.
#
#
# 3 Isolated particle MD
# ======================
#
# When simulating isolated particles with MD, it is sometimes preferable
# to set random momenta corresponding to a specific temperature and let the
# system evolve freely. With a relatively high temperature, the is however
# a risk that the collection of atoms will drift out of the simulation box
# because the randomized momenta gave the center of mass a small but
# non-zero velocity too.
#
# Let us see what happens when we propagate a nanoparticle for a long time:

# %%
from ase import units
from ase.cluster.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.verlet import VelocityVerlet
from ase.optimize import QuasiNewton

use_asap = False

if use_asap:
    from asap3 import EMT

    size = 4
else:
    from ase.calculators.emt import EMT

    size = 2

# Set up a nanoparticle
atoms = FaceCenteredCubic(
    "Cu",
    surfaces=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
    layers=(size, size, size),
    vacuum=4,
)
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
dyn = VelocityVerlet(atoms, 5 * units.fs, trajectory="moldyn4.traj")


def printenergy(a=atoms):
    """Function to print potential, kinetic, and total energy per atom."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print(
        f"Energy per atom: Epot = {epot:6.3f} eV  "
        f"Ekin = {ekin:.3f} eV  "
        f"(T = {ekin / (1.5 * units.kB):3.0f} K)  "
        f"Etot = {epot + ekin:.3f} eV"
    )


printenergy()
dyn.attach(printenergy, interval=10)
dyn.run(2000)

# %%
# After running the simulation, use :ref:`ase-gui` to compare the results
# with how it looks if you comment out either the line that says
# ``Stationary(atoms)``, ``ZeroRotation(atoms)`` or both
#
# ::
#
#   ase gui moldyn4.traj
#
# Try playing the movie with a high frame rate and set frame skipping to a
# low number. Can you spot the subtle difference?
