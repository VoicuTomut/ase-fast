""".. _tipnp water box equilibration:

Equilibrating a TIPnP Water Box
===============================

This tutorial shows how to use the TIP3P and TIP4P force fields in
ASE."""
# %%
# .. note::
#
#   Due to GitLab limit we have to cut the simulation steps in the tutorial.
#   Please adjust the number of steps to your own system.
#   We recommend multiply the number of steps we provided by 1000 for
#   a realistic use.

# %%
import ase.units as units
from ase.build import molecule
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths
from ase.io.trajectory import Trajectory
from ase.md import Langevin

# %%
# Let's first create a starting point of the simulaiton.
# We will create a water box at 20 °C density.
atoms = molecule('H2O')
density = 0.9982  # denisty of water in g/cm^3 at 20°C
box_length = ((atoms.get_masses().sum() / units.mol) / (density * 1e-24)) ** (
    1 / 3
)  # box length in Å
atoms.set_cell((box_length, box_length, box_length))
atoms.center()
# Repeat the water molecule we just created to end up with a PBC cell
atoms *= (3, 3, 3)
atoms.set_pbc(True)

# %%
# We can visualise the starting box
import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

fig, ax = plt.subplots()
plot_atoms(atoms, ax)
ax.set_axis_off()
# %%
# Since the TIPnP type water interpotentials are for rigid
# molecules, there are no intramolecular force terms, and we need to
# constrain all internal degrees of freedom. For this, we're
# using the RATTLE-type constraints of the :ref:`FixBondLengths` class to
# constrain all internal atomic distances (O-H1, O-H2, and H1-H2) for
# each molecule.
atoms.constraints = FixBondLengths(
    [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(3**3) for j in [0, 1, 2]]
)
# RATTLE-type constraints on O-H1, O-H2, H1-H2.
tag = 'tip3p_27mol_equil'
atoms.calc = TIP3P(rc=4.5)  # set the calculator to be the TIP3P force field

# %%
# For efficiency, we first equillibrate a smaller box, and then repeat that
# once more for the final equillibration. However, the potentials are not
# parallelized, and are mainly included for testing and for use with QM/MM
# tasks, so expect to let it run for some time.
# For illustration, we will equillibrate our system with the Langevin
# thermostat.

# Equillibrate in a small box first.
md = Langevin(
    atoms,
    1 * units.fs,
    temperature_K=293.15,  # 20 °C
    friction=0.01,
    logfile=tag + '.log',
)

traj = Trajectory(tag + '.traj', 'w', atoms)
md.attach(traj.write, interval=1)
md.run(4)  # please use 4000 to better equilibrate

# Repeat box and equilibrate further.
tag = 'tip3p_216mol_equil'
atoms.set_constraint()  # repeat not compatible with FixBondLengths currently.
atoms *= (2, 2, 2)
atoms.constraints = FixBondLengths(
    [
        (3 * i + j, 3 * i + (j + 1) % 3)
        for i in range(len(atoms) // 3)
        for j in [0, 1, 2]
    ]
)
atoms.calc = TIP3P(rc=7.0)
md = Langevin(
    atoms,
    2 * units.fs,
    temperature_K=293.15,  # 20 °C
    friction=0.01,
    logfile=tag + '.log',
)

traj = Trajectory(tag + '.traj', 'w', atoms)
md.attach(traj.write, interval=1)
md.run(2)  # please use 2000 to better equilibrate
# %%
# .. note::
#
#  The temperature calculated by ASE is assuming all degrees of freedom
#  are available to the system. Since the constraints have removed the 3
#  vibrational modes from each water, the shown temperature will be 2/3
#  of the actual value.

# %%
# The procedure for the TIP4P force field is the same, with the following
# exception: the atomic sequence **must** be OHH, OHH, ... .
# So to perform the same task using TIP4P, you simply have to import
# that calculator instead:
# ``from ase.calculators.tip4p import TIP4P, rOH, angleHOH``
#
# More info about the TIP4P potential: :mod:`ase.calculators.tip4p`
