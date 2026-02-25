""".. _example_qmmm:

==========================
QM/MM Simulations with ASE
==========================

QM/MM Simulations couple two (or, in principle, more)
descriptions to get total energy
and forces for the entire system in an efficient manner.
ASE has a native Explicit Interaction calculator,
:class:`~ase.calculators.qmmm.EIQMMM`, that uses an electrostatic embedding
model to couple the subsystems explicitly. See
:doi:`the method paper for more info <10.1021/acs.jctc.7b00621>`.

Examples of what this code has been used for can be seen
:doi:`here <10.1021/jz500850s>`,
and :doi:`here <10.1021/acs.inorgchem.6b01840>`.

This section will show you how to setup up various QM/MM simulations.
We will be using GPAW_ for the QM part. Other QM calculators should
be straightforwardly compatible with the subtractive-scheme SimpleQMMM
calculator, but for the Excplicit Interaction EIQMMM calculator, you
would need to be able to put an electrostatic external potential into
the calculator for the QM subsystem. This is often simply a matter of:

1. Making the ASE-calculator write out the positions and charge-values
   to a format that your QM calculator can parse.
2. Read in the forces on the point charges from the QM density.
"""

# %%
# ASE-calculators that currently support EIQMM:
#
# 1. GPAW_
# 2. DFTBplus_
# 3. CRYSTAL_
# 4. TURBOMOLE_
#
# To see examples of how to make point charge potentials for EIQMMM,
# have a look at the :class:`~ase.calculators.dftb.PointChargePotential`
# classes in any of the calculators above.
#
# .. _GPAW: https://gpaw.readthedocs.io/
# .. _DFTBplus: https://ase-lib.org/ase/calculators/dftb.html
# .. _CRYSTAL: https://ase-lib.org/ase/calculators/crystal.html
# .. _TURBOMOLE: https://ase-lib.org/ase/calculators/turbomole.html
#
# You might also be interested in the solvent MM potentials included in ASE.
# The tutorial on :ref:`tipnp water box equilibration` could be relevant to
# have a look at. For acetonitrile, have a look at :ref:`acn_md_tutorial`.
#
# Some MD codes have more advanced solvators, such as AMBER_, and stand-alone
# programs such as PACKMOL_ might also come in handy.
#
# .. _AMBER: https://ambermd.org/AmberTools.php
# .. _PACKMOL: http://m3g.iqm.unicamp.br/packmol/home.shtml
#
#
# General Workflow
# ----------------
# The implementation was developed with the focus of
# modelling ions and complexes
# in solutions, we're working on expanding its functionality to encompass
# surfaces.
#
# In broad strokes, the steps to performing QM/MM MD simulations for thermal
# sampling or dynamics studies, these are the steps:
#
# QM/MM MD General Strategy for A QM complex in an MM solvent:
#
# 1. Equilibrate an MM solvent box using one of the MM potentials built into
#    ASE (see :ref:`tipnp water box equilibration` for water potentials), one
#    of the compatible external MM codes, or write your own potential
#    (see :ref:`Adding new calculators`)
# 2. Optimize the gas-phase structure of your QM complex in GPAW, analyze what
#    level of accuracy you will need for your task.
# 3. Place the relaxed structure of the QM molecule in your MM solvent box,
#    deleting overlapping MM molecules.
# 4. Re-equillibrate the QM/MM system.
# 5. Run production runs.

# %%
# Electrostatic Embedding QM/MM
# -----------------------------
# Theoretical Background
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The total energy expression for the full QM/MM system is:
#
# .. math::  E_\mathrm{TOT} = E_\mathrm{QM} + E_\mathrm{I} + E_\mathrm{MM}.
#
# The MM region is modelled using point charge force fields, with charges
# :math:`q_i` and :math:`\tau_i` denoting their spatial coordinates, so the
# QM/MM coupling term :math:`E_\mathrm{I}` will be
#
# .. math:: E_\mathrm{I} = \sum_{i=1}^C q_i \int \frac{n({\bf r})}{\mid\!{\bf r}
#                          - \tau_i\!\mid}\mathrm{d}{\bf r} +
#                          \sum_{i=1}^C\sum_{\alpha=1}^A
#                          \frac{q_i Z_{\alpha}}{\mid\!{\bf R}_\alpha
#                          - \tau_i\!\mid} + E_\mathrm{RD}
#
# where :math:`n({\bf r})` is the spatial electronic density of the quantum
# region, :math:`Z_\alpha` and :math:`{\bf R}_\alpha` are the charge and
# coordinates of the nuclei in the QM region, respectively, and
# :math:`E_\mathrm{RD}` is the term describing the remaining, non-Coulomb
# interactions between the two subsystems.
#
# For the MM point-charge external potential in GPAW, we use the total pseudo-
# charge density :math:`\tilde{\rho}({\bf r})` for the coupling, and since the
# Coloumb integral is evaluated numerically on the real space grid, thus the
# coupling term ends up like this:
#
# .. math:: E_\mathrm{I} = \sum_{i=1}^C q_i \sum_{g}
#   \frac{\tilde{\rho}({\bf r})}{\mid\!{\bf r}_g  - \tau_i\!\mid} v_g
#   + E_\mathrm{RD}
#
# Currently, the term for :math:`E_{\mathrm{RD}}` implemented is a Lennard-
# Jones-type potential:
#
# .. math:: E_\mathrm{RD} = \sum_i^C \sum_\alpha^A
#                           4\epsilon\left[
#                           \left(\frac{\sigma}{\mid\!{\bf R}_\alpha
#                           - \tau_i\!\mid}\right)^{12}
#                           - \left(\frac{\sigma}{\mid\!{\bf R}_\alpha
#                           - \tau_i\!\mid}\right)^{6} \right]
#
# Example: QM/MM for a Water Dimer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's first do a very simple electrostatic embedding QM/MM single point
# energy calculation on the water dimer. The necessary inputs are described in
# the :class:`ase.calculators.qmmm.EIQMMM` class.
#
#
# The following script will calculate the QM/MM single point energy of the
# water dimer from the :ref:`s22`, using LDA and TIP3P,
# for illustration purposes.

import matplotlib.pyplot as plt
import numpy as np
from gpaw import GPAW

from ase.calculators.qmmm import EIQMMM, Embedding, LJInteractions
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
from ase.data import s22
from ase.visualize.plot import plot_atoms

# Create system
atoms = s22.create_s22_system('Water_dimer')
atoms.center(vacuum=4.0)

# visualization
fig, ax = plt.subplots()
plot_atoms(atoms, ax=ax)
ax.set_axis_off()

# Make QM atoms selection of first water molecule:
qm_idx = range(3)

# Set up interaction & embedding object
interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})
embedding = Embedding(rc=0.02)  # Short range analytical potential cutoff

# Set up calculator
atoms.calc = EIQMMM(
    qm_idx,
    GPAW(txt='qm.out', mode='fd', xc='LDA'),
    TIP3P(),
    interaction,
    embedding=embedding,
    vacuum=None,  # if None, QM cell = MM cell
    output='qmmm.log',
)

print(f'The potential energy of the system is {atoms.get_potential_energy()}')

# %%
# Here, we have just used the TIP3P LJ parameters for the QM part as well. If
# this is a good idea or not isn't trivial. The LJInteractions module needs
# combined parameters for all possible permutations of atom types in your
# system, that have LJ parameters. A list of combination rules can be found
# `here <http://www.sklogwiki.org/SklogWiki/index.php/Combining_rules>`_.
# An overview over different TIP3P LJ parameters can be found
# e.g., on the `LAMMPS website <https://docs.lammps.org/Howto_tip3p.html>`_.
# Here's a code snippet of how to combine LJ parameters of atom types A and B
# via the Lorentz-Berthelot rules:

import itertools as it

epsHH = 0.0
sigHH = 1.0
epsOO = epsilon0
sigOO = sigma0

parameters = {'H': (epsHH, sigHH),
              'O': (epsOO, sigOO)}


def lorenz_berthelot(p):
    combined = {}
    for comb in it.product(p.keys(), repeat=2):
        combined[comb] = ((p[comb[0]][0] * p[comb[1]][0])**0.5,
                        (p[comb[0]][1] + p[comb[1]][1]) / 2)
    return combined


combined = lorenz_berthelot(parameters)
interaction = LJInteractions(combined)

# %%
# This will (somewhat redundantly) yield:

print(combined)

# %%
# It is also possible to run structural relaxations and molecular dynamics
# using the electrostatic embedding scheme:

from ase.constraints import FixBondLengths
from ase.optimize import LBFGS

mm_bonds = [(3, 4), (4, 5), (5, 3)]
atoms.constraints = FixBondLengths(mm_bonds)
dyn = LBFGS(atoms=atoms, trajectory='dimer.traj')
# Relaxation (we only relax to very loose fmax here,
# lower values should be used for serious calculations)
dyn.run(fmax=0.5)  # e.g. fmax=0.05

# %%
# Since TIP3P is a rigid potential, we constrain all interatomic distances.
# QM bond lengths can be constrained too, in the same manner.
#
# Periodicity
# ^^^^^^^^^^^
# For these types of simulations with GPAW, you'd probably want two cells:
# a QM (non-periodic) and and MM cell (periodic):

atoms.set_pbc(True)
# Set up calculator
atoms.calc = EIQMMM(
    qm_idx,
    GPAW(txt='qm.out', mode='fd', xc='LDA'),
    TIP3P(),
    interaction,
    embedding=embedding,
    vacuum=4.,  # Now QM cell has walls min. 4 Å from QM atoms
    output='qmmm.log')

# %%
# This will center the QM subsystem in the MM cell. For QM codes with no single
# real-space grid like GPAW, you can still use this to center your QM subsystem,
# and simply disregard the QM cell, or manually center your QM subsystem,
# and leave vacuum as ``None``.
#
# LJInteractionsGeneral - For More Intricate Systems
# --------------------------------------------------
# It often happens that you will have different 'atom types' (an element in a
# specific environment) per element in your system, i.e. you want to assign
# different LJ-parameters to the oxygens of your solute molecule and the oxygens
# of water. This can be done using
# :class:`~ase.calculators.qmmm.LJInteractionsGeneral`, which takes in NumPy
# arrays with sigma and epsilon values for each individual QM and MM atom,
# respectively, and combines them itself, with Lorentz-Berthelot.
# I.e., for our water dimer from before:

from ase.calculators.qmmm import LJInteractionsGeneral
from ase.calculators.tip3p import epsilon0, sigma0

# General LJ interaction object for the 'OHHOHH' water dimer
sigma_mm = np.array([sigma0, 0, 0])  # Hydrogens have 0 LJ parameters
epsilon_mm = np.array([epsilon0, 0, 0])
sigma_qm = np.array([sigma0, 0, 0])
epsilon_qm = np.array([epsilon0, 0, 0])
interaction = LJInteractionsGeneral(sigma_qm, epsilon_qm,
                                    sigma_mm, epsilon_mm,
                                    qm_molecule_size=3,
                                    mm_molecule_size=3)
# %%
# The ``qm_molecule_size`` and ``mm_molecule_size`` should
# be the number of atoms
# per molecule. Often the ``qm_molecule_size`` will
# simply be the total number of
# atoms in your QM subsystem. Here, our MM subsystem is comprised of a single
# water molecule, but say we had N water molecules in the MM subsystem,
# we wouldn't need to repeat e.g. the ``sigma_mm`` array N times,
# we can simply keep it as it is
# written in the above.
#
#
# EIQMMM and Charged Systems - Counterions
# ----------------------------------------
# If your QM subsystem is charged, it is good to charge-neutrialize the entire
# system. This can be done in ASE by adding MM 'counterions', which are simple,
# single-atomic particles that carry a charge,
# and interact with the solvent and solute
# through a Coulomb and an LJ term.
# The implementation is rather simplified and should only
# serve to neutralize the total system. It might be a good idea to restrain them
# so they do not diffuse too close to the QM subsystem,
# as the effective concentration
# in your simulation cell might be a lot higher than in real life.
#
# Example: QM/MM for a NaCl in Water
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As simple example we add here two NaCl to our system,
# where we treat the 2 Na as part of the QM region and the Cl as part
# of the MM region.

from ase import Atoms

ions = Atoms('Na2Cl2', positions=[[1, 4, 3], [2, 8, 6], [10, 4, 2], [7, 8, 6]])
atoms += ions

# resort so that the regions are together
atoms = Atoms([atoms[i] for i in [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]])

na_idx = [i.index for i in atoms if i.number == 11]
qm_idx = list(qm_idx) + na_idx

# visualization
fig, ax = plt.subplots()
plot_atoms(atoms, ax=ax)
ax.set_axis_off()

# %%
# To use the implementation, you need to 'Combine'
# two MM calculators, one for the
# counterions, and one for your solvent. This is an example of combining two Cl-
# ions with TIP3P water, using the TIP3P LJ-arrays from the previous section:

from ase import units
from ase.calculators.combine_mm import CombineMM
from ase.calculators.counterions import AtomicCounterIon as ACI

# Cl-:  10.1021/ct600252r
sigCl = 4.02
epsCl = 0.71 * units.kcal / units.mol

# in this sub-atoms object, CombineMM only sees Cl and Water,
# and Cl is here atom 3 and 4
mmcalc = CombineMM([3, 4],  # indices of the counterion atoms in the MM region
                   apm1=1, apm2=3,  # atoms per 'molecule' of each subgroup
                   calc1=ACI(-1, epsCl, sigCl),  # Counterion calculator
                   calc2=TIP3P(),  # Water calculator
                   sig1=[sigCl], eps1=[epsCl],  # LJ Params for subgroup1
                   sig2=sigma_mm, eps2=epsilon_mm)  # LJ params for subgroup2

atoms_mm = Atoms([i for i in atoms if i.index > 2 and i.number != 11])
print(atoms_mm)
atoms_mm.calc = mmcalc
print('MM', atoms_mm.get_potential_energy())


# %%
# The charge of the counterions is defined as -1 in the first
# input in ``ACI``, which
# Then also takes LJ-parameters for interactions with other ions
# beloning to this calculator.
#
# This ``mmcalc`` object is then used in the initialization
# of the ``EIQMMM`` calculator.
# But before we can do that, the QM/MM Lennard-Jones potential
# needs to understand that
# The total MM subsystem is now comprised of two subgroups,
# the counterions and the water.
# That is done by initializing the ``interaction`` object with a
# tuple of NumPy arrays for the MM
# Part. So if your QM subsystem has 5 atoms, you'd do:

# Na+: DOI 10.1021/ct600252r
sigNa = 4.07
epsNa = 0.0005 * units.kcal / units.mol

sigma_qm = (np.array([sigma0, 0, 0, sigNa, sigNa]))
epsilon_qm = (np.array([epsilon0, 0, 0, epsNa, epsNa]))
sigma_mm = (np.array([sigma0, 0, 0]), np.array([sigCl]))
epsilon_mm = (np.array([epsilon0, 0, 0]), np.array([epsCl]))

interaction = LJInteractionsGeneral(sigma_qm, epsilon_qm,
                                    sigma_mm, epsilon_mm,
                                    qm_molecule_size=5,
                                    mm_molecule_size=5)


# %%
# Current limitations:
#
# * No QM PBCs
# * There is currently no automated way of doing QM/MM over covalent bonds
#   (inserting cap-atoms, redistributing forces ...)

# %%
# Other tips:
# ___________
#
# If you are using GPAW and water, consider having a look at the
# `much faster RATTLE constraints for water here <https://gitlab.com/gpaw/gpaw/blob/master/gpaw/test/rattle.py>`__
