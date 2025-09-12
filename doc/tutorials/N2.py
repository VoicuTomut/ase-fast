from ase import Atoms
from ase.calculators.emt import EMT

atom = Atoms('N')
atom.calc = EMT()
e_atom = atom.get_potential_energy()

d = 1.1
molecule = Atoms('2N', [(0.0, 0.0, 0.0), (0.0, 0.0, d)])
molecule.calc = EMT()
e_molecule = molecule.get_potential_energy()

e_atomization = e_molecule - 2 * e_atom

print(f'Nitrogen atom energy: {e_atom:5.2f} eV')
print(f'Nitrogen molecule energy: {e_molecule:5.2f} eV')
print('Atomization energy: %5.2f eV' % -e_atomization)
