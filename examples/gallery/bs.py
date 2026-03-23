"""

Band-structure
==============


"""

# creates: cu.png
from ase.build import bulk
from ase.calculators.test import FreeElectrons

atoms = bulk('Cu')
atoms.calc = FreeElectrons(nvalence=1, kpts={'path': 'GXWLGK', 'npoints': 200})
atoms.get_potential_energy()
bs = atoms.calc.band_structure()
bs.plot(emin=0, emax=20, filename='cu.png')
