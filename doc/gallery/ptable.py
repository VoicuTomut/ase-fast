# creates: ptable.png
from ase.utils.ptable import ptable

atoms = ptable()
atoms = ptable(spacing = 4.0)

atoms.write('ptable.png')
atoms.write('ptable.eps')
atoms.write('ptable_pov.pov') # Calling "povray ptable_pov.ini" will render it with povray.


print('xyz positions',atoms.positions)
print('xyz position ranges',atoms.positions.max(axis=0)- atoms.positions.min(axis=0))
#atoms.write('ptable.pdf')
#   from ase.visualize import view
#   view(atoms)
