# creates: ptable.png
from ase.utils.ptable import ptable

atoms = ptable()
atoms.write('ptable.png')

# Calling "povray ptable_pov.ini" will render it with povray.
# atoms.write('ptable_pov.pov')

#atoms.write('ptable.pdf')
#   from ase.visualize import view
#   view(atoms)
