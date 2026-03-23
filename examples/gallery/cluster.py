"""

Nanoparticle
============

"""

from ase.cluster.cubic import FaceCenteredCubic
from ase.visualize.plot import plot_atoms

surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [6, 9, 5]
lc = 3.61000
culayer = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc)
culayer.rotate(6, 'x', rotate_cell=True)
culayer.rotate(2, 'y', rotate_cell=True)
plot_atoms(culayer)

surfaces = [(1, 0, 0), (1, 1, 1), (1, -1, 1)]
layers = [6, 5, -1]
trunc = FaceCenteredCubic('Cu', surfaces, layers)
trunc.rotate(6, 'x', rotate_cell=True)
trunc.rotate(2, 'y', rotate_cell=True)
plot_atoms(trunc)
