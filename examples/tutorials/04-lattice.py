""".. _lattice_constant_example:

Finding lattice constants using EOS and the stress tensor
=========================================================

"""

# %%
# Introduction
# ============
# The bulk lattice constant for a material modelled with DFT is often different
# from the experimental lattice constant. For example, for DFT-GGA functionals,
# lattice constants can be on the order of 4 % larger than the experimental
# value. To model materials' properties accurately, it is important to use
# the optimized lattice constant corresponding the method/functional used.
#
# In this tutorial, we will first look at obtaining the lattice constant by
# fitting the equation of state and then obtaining it from the stress tensor.

# %%
# Finding lattice constants using the equation of state
# -----------------------------------------------------
#

# %%
# FCC
# ^^^
# The lattice constant :math:`a` for FCC bulk metal can be obtained using the
# equation of state as outline in the tutorial :ref:`eos_example` by calculating
# :math:`a^3 = V`, where :math:`V` is the volume of the unit cell.

# %%
# HCP
# ^^^
# For HCP bulk metals, we need to account for two lattice constants, :math:`a`
# and :math:`c`. Let's try to find :math:`a` and :math:`c` for HCP nickel
# using the :mod:`EMT <ase.calculators.emt>` potential.
#
# First, we make an initial guess for :math:`a` and :math:`c` using the FCC
# nearest neighbor distance and the ideal :math:`c/a` ratio:

import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import StrainFilter  # Import for stress tensor calculation
from ase.io import Trajectory, read
from ase.optimize import BFGS  # Import for stress tensor calculation

a0 = 3.52 / np.sqrt(2)
c0 = np.sqrt(8 / 3.0) * a0

# %%
# We create a trajectory to save the results:

traj = Trajectory('Ni.traj', 'w')

# %%
# Next, we do the 9 calculations (three values for :math:`a` and three for
# :math:`c`). Note that for a real-world case, we would want to try more values.
eps = 0.01
for a in a0 * np.linspace(1 - eps, 1 + eps, 3):
    for c in c0 * np.linspace(1 - eps, 1 + eps, 3):
        ni = bulk('Ni', 'hcp', a=a, c=c)
        ni.calc = EMT()
        ni.get_potential_energy()
        traj.write(ni)

# %%
# Analysis
# --------
# Now, we need to extract the data from the trajectory.  Try this:
ni = bulk('Ni', 'hcp', a=2.5, c=4.0)
print(ni.cell)

# %%
# So, we can get :math:`a` and :math:`c` from ``ni.cell[0, 0]``
# and ``ni.cell[2, 2]``:
configs = read('Ni.traj@:')
energies = [config.get_potential_energy() for config in configs]
a = np.array([config.cell[0, 0] for config in configs])
c = np.array([config.cell[2, 2] for config in configs])

# %%
# We fit the energy :math:`E` to an expression for the equation of state,
# e.g., a polynomial:
#
# .. math:: E = p_0 + p_1 a + p_2 c + p_3 a^2 + p_4 ac + p_5 c^2
#
# :math:`p_i` are the parameters. We can find the parameters for the
# best fit, e.g., through least squares fitting:
functions = np.array([a**0, a, c, a**2, a * c, c**2])
p = np.linalg.lstsq(functions.T, energies, rcond=-1)[0]

# %%
# then, we can find the minimum like this:
p0 = p[0]
p1 = p[1:3]
p2 = np.array([(2 * p[3], p[4]), (p[4], 2 * p[5])])
a0, c0 = np.linalg.solve(p2.T, -p1)

# Save the lattice constants to a file
with open('lattice_constant.csv', 'w') as fd:
    fd.write(f'{a0:.3f}, {c0:.3f}\n')

# Show the results on screen:
print('The optimized lattice constants are:')
print(f'a = {a0:.3f} Å, c = {c0:.3f} Å')

# %%
# Using the stress tensor
# =======================
#
# One can also use the stress tensor to optimize the unit cell.
# Using 'StrainFilter', the unit cell is relaxed until the stress is
# below a given threshold.
#
# Note that if the initial guesses for :math:`a` and :math:`c` are far
# from the optimal values, the optimization can get stuck in a local minimum.
# Similarly, if the threshold is not chosen tight enough, the resulting
# lattice constants can again be far from the optimal values.

ni = bulk('Ni', 'hcp', a=3.0, c=5.0)
ni.calc = EMT()

# %% Note that the optimizer is applied on top of the StrainFilter,
# rather then directly to the atoms. The filter presents an
# alternative view of the atomic degrees of freedom: instead of
# modifying atomic positions to minimise target forces, the `BFGS
# <https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm>`_
# algorithm is allowed to modify lattice parameters to minimise target
# stress.

sf = StrainFilter(ni)
opt = BFGS(sf)

# If you want the optimization path in a trajectory, comment in these lines:
# traj = Trajectory('path.traj', 'w', ni)
# opt.attach(traj)

# Set the threshold and run the optimization::
opt.run(0.005)  # run until forces are below 0.005 eV/Å³

# %%
# Analyze the results
# print the unit cell
print('The optimized lattice constants from the stress tensor are:')
print(f'a = {ni.cell[0, 0]:.3f} Å, c = {ni.cell[2, 2]:.3f} Å')
