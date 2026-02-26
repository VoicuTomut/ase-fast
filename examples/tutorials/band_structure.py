""".. _band_structure:

Band Structures of Bulk Structures
==================================

Here, we calculate the band structure of relaxed bulk crystal structures.
"""

# %%

import matplotlib.pyplot as plt
from gpaw import GPAW, PW

from ase.build import bulk
from ase.dft.dos import DOS

# %%
#
# Setting up bulk structures
# --------------------------
#
# For more details regarding the set-up and relaxation of bulk structures,
# check out the :ref:`bulk` tutorial.

atoms = bulk('Ag')

# %%
#
# Bulk DFT calculation
# --------------------
#
# For periodic DFT calculations we should generally use a number of
# k-points which properly samples the Brillouin zone.
# Many calculators including GPAW and Aims
# accept the ``kpts`` keyword which can be a tuple such as
# ``(4, 4, 4)``.  In GPAW, the planewave mode
# is very well suited for smaller periodic systems.
# Using the planewave mode, we should also set a planewave cutoff (in eV):
#

calc = GPAW(
    mode=PW(350), kpts=[8, 8, 8], txt='gpaw.bulk_Ag.txt', setups={'Ag': '11'}
)

# %%
# Here we have used the ``setups`` keyword to specify that we want the
# 11-electron PAW dataset instead of the default which has 17 electrons,
# making the calculation faster.
#
# (In principle, we should be sure to converge both kpoint sampling
# and planewave cutoff -- I.e., write a loop and try different samplings
# so we know both are good enough to accurately describe the quantity
# we want.)

atoms.calc = calc
print(
    'Bulk {0} potential energy = {1:.3f}eV'.format(
        atoms.get_chemical_formula(), atoms.get_potential_energy()
    )
)

# %%
# We can save the ground-state into a file

ground_state_file = 'bulk_Ag_groundstate.gpw'
calc.write(ground_state_file)

# %%
#
# Density of states
# -----------------
#
# Having saved the ground-state, we can reload it for ASE to extract
# the density of states:
#

calc = GPAW(ground_state_file)
dos = DOS(calc, npts=800, width=0)
energies = dos.get_energies()
weights = dos.get_dos()

# %%
# Calling the DOS class with ``width=0`` means ASE calculates the DOS using
# the linear tetrahedron interpolation method, which takes time but gives a
# nicer representation. If the width is nonzero (e.g., 0.1 (eV)) ASE uses a
# simple Gaussian smearing with that width, but we would need more k-points
# to get a plot of the same quality.
#

fig, ax = plt.subplots()
ax.axvline(0.0, linestyle='--', color='black', alpha=0.5)
ax.plot(energies, weights)
ax.set_xlabel('Energy - Fermi Energy (eV)')
ax.set_ylabel('Density of States (1/eV)')
fig.tight_layout()

# %%
#
# Time for analysis: Which parts of the spectrum do you think originate
# (mostly) from s electrons?  And which parts (mostly) from d electrons?
#
# As we probably know, the d-orbitals in a transition metal atom are
# localized close to the nucleus while the s-electron is much more
# delocalized.
#
# In bulk systems, the s-states overlap a lot and therefore split into a
# very broad band over a wide energy range.  d-states overlap much less
# and therefore also split less:  They form a narrow band with a
# very high DOS.  Very high indeed because there are 10 times as
# many d electrons as there are s electrons.
#
# So to answer the question, the d-band accounts for most of the states
# forming the big, narrow chunk between -6.2 eV to -2.6 eV.  Anything outside
# that interval is due to the much broader s band.
#
# The DOS above the Fermi level may not be correct, since the SCF
# convergence criterion (in this calculation)
# only tracks the convergenece of occupied states.
# Hence, the energies over the Fermi level 0 are probably wrong.
#
#
# What characterizes the noble metals Cu, Ag, and Au, is that the d-band
# is fully occupied.  I.e.: The whole d-band lies below the Fermi level
# (energy=0).
# If we had calculated any other transition metal, the Fermi level would
# lie somewhere within the d-band.
#
# .. note::
#
#    We could calculate the s, p, and d-projected DOS to see more
#    conclusively which states have what character.
#    In that case we should look up the GPAW documentation, or other
#    calculator-specific documentation.  So let's not do that now.
#

# %%
# Band structure
# --------------
#
# Let's calculate the band structure of silver.
#
# First we need to set up a band path.  Our favourite image search
# engine can show us some reference graphs.  We might find band
# structures from both Exciting and GPAW with Brillouin-zone path:
#
# :math:`\mathrm{W L \Gamma X W K}`.
#
# Luckily ASE knows these letters and can also help us
# visualize the reciprocal cell:
#

lat = atoms.cell.get_bravais_lattice()
print(lat.description())

# %%

lat.plot_bz(show=True)
plt.show()

# %%
#
# In general, the :mod:`ase.lattice` module provides
# :class:`~ase.lattice.BravaisLattice` classes used to represent each
# of the 14 + 5 Bravais lattices in 3D and 2D, respectively.
# These classes know about the high-symmetry k-points
# and standard Brillouin-zone paths
# (using the `AFlow <http://aflowlib.org/>`_ conventions).
# You can visualize the band path object in bash via the command:
#
#   $ ase reciprocal path.json
#
# You can ``print()`` the band path object to see some basic information
# about it, or use its :meth:`~ase.dft.kpoints.BandPath.write` method
# to save the band path to a json file such as :file:`path.json`.
#

path = atoms.cell.bandpath('WLGXWK', density=10)
path.write('path.json')
print(path)

# %%

path.plot()
plt.show()

# %%
# Once we are sure we have a good path with a reasonable number of k-points,
# we can run the band structure calculation.
# How to trigger a band structure calculation depends
# on which calculator we are using, so we would typically consult
# the documentation for that calculator (ASE will one day provide
# shortcuts to make this easier with common calculators):
#

calc = GPAW(ground_state_file)
calc = calc.fixed_density(kpts=path, symmetry='off')

# %%
# We have here told GPAW to use our bandpath for k-points, not to
# perform symmetry-reduction of the k-points, and to fix the electron
# density.
#
# Then we trigger a new calculation, which will be non-selfconsistent,
# and extract and save the band structure:
#

bs = calc.band_structure()
bs.write('bs.json')
print(bs)

# %%
# We can plot the band structure in python, or in the terminal
# from a file using:
#
#   $ ase band-structure bs.json
#
# The plot will show the Fermi level as a dotted line
# (but does not define it as zero like the DOS plot before).
# Looking at the band structure, we see the complex tangle of what must
# be mostly d-states from before, as well as the few states with lower energy
# (at the :math:`\Gamma` point) and higher energy (crossing the Fermi level)
# attributed to s.
#

ax = bs.plot()
ax.set_ylim(-2.0, 30.0)
plt.show()
