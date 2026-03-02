""".. _defects:

==============================
Defect calculations: ASE Tools
==============================

This section gives an (incomplete) overview of features in ASE that
help in the preparation and analysis of supercell calculations as most
commonly employed in the computation of defect properties.

.. contents::
"""
# %%
# Supercell creation
# ==================
#
# Background
# ----------
#
# Defect properties are most commonly investigated in the so-called
# dilute limit, i.e. under conditions, in which defect-defect
# interactions are negligible. While alternative approaches in
# particular embedding techniques exist, the most common approach is to
# use supercells. To this end, one creates a supercell by a *suitable*
# (see below) repetition of the primitive unit cell, after which a
# defect, e.g., a vacancy or an impurity atom, is inserted.
#
# The calculation thus corresponds to a periodic arrangement of
# defects. Accordingly, care must be taken to keep the interactions
# between defects as small as possible, which generally calls for large
# supercells. Thus the typical goal for generating the simulation supercell
# for defect calculations is to maximize the defect-defect separation in
# *all* directions, for a reasonable number of atoms (and thus computational
# cost). In principle, we can do a good job of this by using a supercell
# with a suitable shape.
# To illustrate this for different lattices, we build and plot
# three 2D lattices with identical
# unit cell area but
# different lattice symmetry. We build and visualize (3,3,1) supercells
# by repeating the unit cell.

import matplotlib.pyplot as plt

from ase.build import bulk
from ase.geometry import get_distances
from ase.visualize.plot import plot_atoms


def add_decoration(ax, conf, centralidx, neighborcutoff, arrow_offset=(0, 0)):
    # function for adding arrows to neighboring atoms
    vectors, distances = get_distances(
        conf.positions[centralidx], p2=conf.positions
    )
    neighboridx = [
        i for i, j in enumerate(distances[0]) if (neighborcutoff >= j)
    ]

    # we normalize the size of the arrows for illustration purposes
    norm = neighborcutoff / 5
    for idx in neighboridx:
        ax.arrow(
            conf.positions[centralidx][0] + arrow_offset[0],
            conf.positions[centralidx][1] + arrow_offset[1],
            vectors[0][idx][0] / norm,
            vectors[0][idx][1] / norm,
            width=0.1,
            color='k',
        )


lattice_constant = 8
centralidx = 4
neighborcutoff = 8


# square lattice
# build structure
conf = bulk('Po', a=lattice_constant)
# moving the atom in the middle of the cell
positions = conf.get_positions()
positions += conf.cell[0][0] / 2
conf.set_positions(positions)
conf = conf.repeat((3, 3, 1))

# plot cs structure
fig, ax = plt.subplots()
add_decoration(ax, conf, centralidx, neighborcutoff, arrow_offset=(0.5, 0.5))
plot_atoms(conf, ax, offset=(0, 0))  # , rotation=('-80x,0y,0z'))
ax.set_axis_off()
ax.text(
    0.1,
    -0.1,
    'square lattice: r$_1$=a, Z$_1$=4',
    transform=ax.transAxes,
    fontsize=16,
)
plt.show()


# rectangular lattice
# build structure
conf = bulk(
    'C',
    crystalstructure='orthorhombic',
    a=lattice_constant,
    b=lattice_constant / 2,
    c=lattice_constant / 2,
)

# moving the atom in the middle of the cell
positions = conf.get_positions()
positions += [conf.cell[0][0] / 2, conf.cell[0][0] / 4, conf.cell[0][0] / 4]
conf.set_positions(positions)
conf = conf.repeat((3, 3, 1))

# plot orc structure
fig, ax = plt.subplots()
add_decoration(ax, conf, centralidx, neighborcutoff, arrow_offset=(0.5, 0.5))
plot_atoms(conf, ax, offset=(0, 0))
ax.set_axis_off()
ax.text(
    0,
    -0.2,
    'rectangular lattice with a 2:1 aspect ratio:\n r$_1$=a/2, Z$_1$=2',
    transform=ax.transAxes,
    fontsize=16,
)
plt.show()

# hexagonal lattice
# build structure
conf = bulk('Be', a=lattice_constant)
# here, we slice the cell to have one a 2D layer of atoms
confmask = [i.index for i in conf if i.position[2] < 1]
conf = conf[confmask]
conf = conf.repeat((3, 3, 1))
positions = conf.get_positions()
positions -= conf.positions[0]
conf.set_positions(positions)

# plot hpc structure
fig, ax = plt.subplots()
add_decoration(
    ax,
    conf,
    centralidx,
    neighborcutoff,
    arrow_offset=(-conf.cell[1][0] + 1, 1.5),
)
plot_atoms(conf, ax, offset=(0, 0), rotation=('0x,0y,0z'))
ax.set_axis_off()
ax.text(
    0.1,
    -0.1,
    'hexagonal lattice: r$_1$=1.075a, Z$_1$=6',
    transform=ax.transAxes,
    fontsize=16,
)
plt.show()


# %%
#
# In the case of the square lattice, each defect has :math:`Z_1=4`
# nearest neighbors at a distance of :math:`r_1=a_0`, where
# :math:`a_0=\sqrt{A}` with :math:`A` being the unit cell area. By
# comparison in a rectangular lattice with an aspect ratio of 2:1, the
# defects are much closer to each other with :math:`r_1 = a_0/\sqrt{2}` and
# :math:`Z_1=2`, where again :math:`a_0` = :math:`\sqrt{A}`
# (the 'effective cubic length').
# The largest defect-defect distance (at constant unit
# cell area) is obtained for the hexagonal lattice, which also
# correponds to the most closely packed 2D arrangement. Here, one
# obtains :math:`r_1=\sqrt{2}/\sqrt[4]{3}=1.075 a_0` and
# :math:`Z_1=6`. For defect calculations, supercells corresponding to
# hexagonal or square lattices have thus clear advantages. This argument
# can be extended to 3D: Square lattices in 2D correspond to cubic
# lattices (supercells) in 3D with :math:`r_1=a_0` and
# :math:`Z_1=6`. The 3D analogue of the hexagonal 2D lattice are
# hexagonal and cubic close packed structures (i.e. FCC, HCP), both of which
# yield :math:`r_1 = \sqrt[6]{2} a_0 \approx 1.1225 a_0` and :math:`Z_1=12`.
#
# It is straightforward to construct cubic or face-centered cubic (fcc,
# cubic closed packed) supercells for cubic materials (including e.g,
# diamond and zincblende) by using simple repetitions of the
# conventional or primitive unit cells. For countless materials of lower
# symmetry the choice of a supercell is, however not necessarily so
# simple. The algorithm below represents a general solution to this
# issue.
#
# In the case of semiconductors and insulators with small dielectric
# constants, defect-defect interactions are particularly pronounced due
# to the weak screening of long-ranged electrostatic interactions. While
# various correction schemes have been proposed, the most reliable
# approach is still finite-size extrapolation using supercells of
# different size. In this case care must be taken to use a sequence of
# self-similar supercells in order for the extrapolation to be
# meaningful. To motivate this statement consider that the leading
# (monopole-monopole) term :math:`E_{mp}`, which scales with :math:`1/r`
# and is proportional to the (ionic) dielectric constant
# :math:`\epsilon_0`. The :math:`E_{mp}` term is geometry dependent and
# in the case of simple lattices the dependence is easily expressed by
# the Madelung constant. The geometry dependence implies that different
# (super)cell shapes fall on different lines when plotting e.g., the
# formation energy as a function of :math:`N^{-1/3}` (equivalent to an
# effective inverse cell size, :math:`L^{-1} \propto N^{-1/3}`. For
# extrapolation one should therefore only use geometrically equivalent
# cells or at least cells that are as self-similar to each other as
# possibly, see Fig. 10 in [Erhart]_ for a very clear example. In this
# context there is therefore also a particular need for supercells
# of a particular shape.
#
#
# Algorithm for finding optimal supercell shapes
# ----------------------------------------------
#
# The above considerations illustrate the need for a more systematic
# approach to supercell construction. A simple scheme to construct
# "optimal" supercells is described in [Erhart]_. Optimality here
# implies that one identifies the supercell that for a given size
# (number of atoms) most closely approximates the desired shape, most
# commonly a simple cubic or fcc metric (see above). This approach
# ensures that the defect separation is large and that the electrostatic
# interactions exhibit a systematic scaling.
#
# The ideal cubic cell metric for a given volume :math:`\Omega` is simply
# given by :math:`\Omega^{1/3} \mathbf{I}`, which in general does not
# satisfy the crystallographic boundary conditions. The :math:`l_2`-norm
# provides a convenient measure of the deviation of any other cell
# metric from a cubic shape. The optimality measure can thus be defined
# as
#
# .. math::
#   \Delta_\text{sc}(\mathbf{h}) = ||\mathbf{h} - \Omega^{1/3} \mathbf{1}||_2,
#
# Any cell metric that is compatible with the crystal symmetry can be
# written in the form
#
# .. math:: \mathbf{h} = \mathbf{P} \mathbf{h}_p
#
# where :math:`\mathbf{P} \in \mathbb{Z}^{3\times3}` and
# :math:`\mathbf{h}_p` is the primitive cell metric.  This approach can
# be readily generalized to arbitrary target cell metrics. In order to
# obtain a measure that is size-independent it is furthermore convenient
# to introduce a normalization, which leads to the expression
# implemented here, namely
#
# .. math::
#   \bar{\Delta}(\mathbf{Ph}_p) =
#   ||Q\mathbf{Ph}_p - \mathbf{h}_\text{target}||_2,
#
# where `Q = \left(\det\mathbf{h}_\text{target} \big/
# \det\mathbf{h}_p\right)^{1/3}` is a normalization factor.  The
# matrix :math:`\mathbf{P}_\text{opt}` that yields the optimal cell
# shape for a given cell size can then be obtained by
#
# .. math:: \mathbf{P}_\text{opt} =
#   \underset{\mathbf{P}}{\operatorname{argmin}}
#   \left\{ \bar\Delta\left(\mathbf{Ph}_p\right) | \det\mathbf{P}
#   = N_{uc}\right\},
#
# where :math:`N_{uc}` defines the size of the supercell in terms of the
# number of primitive unit cells.
#
#
# Implementation of algorithm
# ---------------------------
#
# For illustration consider the following example. First we set up a
# primitive face-centered cubic (fcc) unit cell and visualize it.

import matplotlib.pyplot as plt

from ase.build import bulk
from ase.visualize.plot import plot_atoms

conf = bulk('Au')


fig, ax = plt.subplots()
plot_atoms(conf, ax)
ax.set_axis_off()

# %%
# Then, we call
# :func:`~ase.build.find_optimal_cell_shape` to obtain a
# :math:`\mathbf{P}` matrix that will enable us to generate a supercell
# with 32 atoms that is as close as possible to a simple cubic shape:
import numpy as np

from ase.build import find_optimal_cell_shape
from ase.build.supercells import eval_length_deviation

P1 = find_optimal_cell_shape(conf.cell, 32, 'sc')
print(P1)

# %%
#
# More nicely rendered, this yields
#
# .. math:: \mathbf{P}_1 =
#   \left(\begin{array}{rrr} -2 & 2 & 2 \\ 2 & -2 & 2 \\
#   2 & 2 & -2 \end{array}\right) \quad
#   \mathbf{h}_1 = \left(\begin{array}{ccc} 2 a_0 & 0 & 0 \\
#   0 & 2 a_0 & 0 \\ 0 & 0 & 2 a_0 \end{array}\right),
#
# where :math:`a_0` =4.05 Å is the lattice constant. This is indeed the
# expected outcome as it corresponds to a :math:`2\times2\times2`
# repetition of the *conventional* (4-atom) unit cell. On the other hand
# repeating this exercise with:

P2 = find_optimal_cell_shape(conf.cell, 495, 'sc')
print(P2)

# %%
# yields a less obvious result, namely
#
# .. math:: \mathbf{P}_2 =
#   \left(\begin{array}{rrr} -6 & 5 & 5 \\ 5 & -6 & 5 \\
#   5 & 5 & -5 \end{array}\right)
#   \quad
#   \mathbf{h}_2 = a_0
#   \left(\begin{array}{ccc} 5 & 0 & 0 \\ 0.5 & 5 & 0.5 \\
#    0.5 & 0.5 & 5 \end{array}\right),
#
# which indeed corresponds to a reasonably cubic cell shape. One can
# also obtain the optimality measure :math:`\bar{\Delta}` by executing:
dev1 = eval_length_deviation(np.dot(P1, conf.cell))
dev2 = eval_length_deviation(np.dot(P2, conf.cell))
print(f'The length deviation for P_1 is {dev1}')
print(f'The length deviation for P_2 is {dev2}')

# %%
# which yields :math:`\bar{\Delta}(\mathbf{P}_1)=0` and
# :math:`\bar{\Delta}(\mathbf{P}_2)=0.0192`.
#
# Since this procedure requires only knowledge of the cell metric (and
# not the atomic positions) for standard metrics, e.g., fcc, bcc, and
# simple cubic one can generate series of shapes that are usable for
# *all* structures with the respective metric. For example the
# :math:`\mathbf{P}_\text{opt}` matrices that optimize the shape of a
# supercell build using a primitive FCC cell are directly applicable to
# diamond and zincblende lattices.
#
# For illustration, the :math:`\bar{\Delta}` values for supercells of SC, FCC
# and BCC lattices with SC/FCC target shapes are shown as a function of
# the number of unit cells :math:`N_{uc}\leq2000` in the panel below (taken
# from :mr:`3404`). The algorithm is, however, most useful for
# non-cubic cell shapes, for which finding several reasonably sized cell
# shapes is more challenging, as illustrated for a hexagonal material
# (LaBr\ :sub:`3`) in [Erhart]_.
#
# .. image:: https://gitlab.com/-/project/470007/uploads/5c52f1b09cfd8f82c3b8453f45762d4f/image.png
#
#
# .. note::
#    For unit cells with more complex space groups,
#    this approach can be cumbersome due
#    to the implementation which loops over many possible
#    transformation matrices. The
#    `find_optimal_cell_shape <https://doped.readthedocs.io/en/latest/doped.utils.html
#    doped.utils.supercells.find_optimal_cell_shape>`_
#    function in `doped <https://doped.readthedocs.io>`_ implements
#    the same algorithm with
#    some efficiency improvements (~100x compute time speedup),
#    and offers an efficient
#    `algorithm <https://doped.readthedocs.io/en/latest/doped.utils.html
#    doped.utils.supercells.find_ideal_supercell>`_
#    for *directly* optimising the periodic defect-defect distance
#    (~10-50% improvements);
#    see [Kavanagh]_ or the ``doped`` `tutorials
#    <https://doped.readthedocs.io/en/latest/generation_tutorial.html>`_.
#
# Generation of supercell
# -----------------------
#
# Once the transformation matrix :math:`\mathbf{P}` it is
# straightforward to generate the actual supercell using e.g., the
# :func:`~ase.build.cut` function. A convenient interface is provided by
# the :func:`~ase.build.make_supercell` function, which is invoked as
# follows:

from ase.build import make_supercell

conf = bulk('Au')
P = find_optimal_cell_shape(conf.cell, 495, 'sc')
supercell = make_supercell(conf, P)

fig, ax = plt.subplots()
plot_atoms(supercell, ax)
ax.set_axis_off()
# %%
#
# .. [Erhart] P. Erhart, B. Sadigh, A. Schleife, and D. Åberg.
#   First-principles study of codoping in lanthanum bromide,
#   Phys. Rev. B, Vol **91**, 165206 (2012),
#   :doi:`10.1103/PhysRevB.91.165206`; Appendix C
#
# .. [Kavanagh] S. R. Kavanagh et al.
#   doped: Python toolkit for robust and repeatable charged defect
#   supercell calculations
#   J. Open Source Softw, 9(**96**), 6433 (2024),
#   :doi:`10.21105/joss.06433`
