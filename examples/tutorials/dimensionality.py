"""
.. _dim_examples:

=======================
Dimensionality analysis
=======================

This is a example of analysis of the dimensionality of a structure using
the :func:`ase.geometry.dimensionality.analyze_dimensionality` function. This is
useful for finding low-dimensional materials, such as 1D chain-like
structures, 2D layered structures, or structures with multiple dimensionality
types, such as 1D+3D.

The example below creates a layered :mol:`MoS_2` structure and analyzes its
dimensionality.
"""

import ase.build
from ase.geometry.dimensionality import analyze_dimensionality

atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
atoms.cell[2, 2] = 7.0
atoms.set_pbc((1, 1, 1))
atoms *= 3

intervals = analyze_dimensionality(atoms, method='RDA')
m = intervals[0]
print(sum([e.score for e in intervals]))
print(m.dimtype, m.h, m.score, m.a, m.b)

atoms.set_tags(m.components)
# Visualize the structure
# from ase.visualize import view
# view(atoms)

# %%
# Coloring the atoms by their tags shows the distinct bonded clusters, which in
# this case are separate layers.
#
# Each component in the material can be extracted, or "*isolated*",
# using the :func:`ase.geometry.dimensionality.isolate_components` function as
# the example below demonstrates.

import numpy as np

import ase.build
from ase import Atoms
from ase.geometry.dimensionality import isolate_components

# build two slabs of different types of MoS2
rep = [4, 4, 1]
a = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19) * rep
b = ase.build.mx2(formula='MoS2', kind='1T', a=3.18, thickness=3.19) * rep
positions = np.concatenate([a.get_positions(), b.get_positions() + [0, 0, 7]])
numbers = np.concatenate([a.numbers, b.numbers])
cell = a.cell
atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[1, 1, 1])
atoms.cell[2, 2] = 14.0

# isolate each component in the whole material
result = isolate_components(atoms)
print('counts:', [(k, len(v)) for k, v in sorted(result.items())])

for dim, components in result.items():
    for atoms in components:
        print(dim)
        # Visualize the structure
        # view(atoms, block=True)

# %%
# The method is described in the article:
#
#  | P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
#  | "Definition of a scoring parameter to identify
#  | low-dimensional materials components"
#  | Phys. Rev. Materials 3 034003, 2019
#  | :doi:`10.1103/PhysRevMaterials.3.034003`
#
# A preprint is available :arxiv:`here <1808.02114>`.
#
# .. seealso::
#
#    More examples here: `Dimensionality analysis of ICSD and COD databases
#    <https://cmr.fysik.dtu.dk/lowdim/lowdim.html>`_.
#
#
# .. autofunction:: ase.geometry.dimensionality.analyze_dimensionality
#   :noindex:
# .. autofunction:: ase.geometry.dimensionality.isolate_components
#   :noindex:
