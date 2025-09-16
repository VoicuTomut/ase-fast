""".. _surface:

Introduction: Nitrogen on copper
================================

This section gives a quick (and incomplete) overview of what ASE can do.

We will calculate the adsorption energy of a nitrogen
molecule on a copper surface.
This is done by calculating the total
energy for the isolated slab and for the isolated molecule. The
adsorbate is then added to the slab and relaxed, and the total energy
for this composite system is calculated. The adsorption energy is
obtained as the sum of the isolated energies minus the energy of the
composite system.

Here is a picture of the system after the relaxation:

.. image:: ../surface.png

Please have a look at the following script :download:`N2Cu.py`:

.. literalinclude:: N2Cu.py

Assuming you have ASE setup correctly (:ref:`download_and_install`)
run the script::

  python N2Cu.py

Please read below what the script does.

Atoms
-----

The :class:`~ase.Atoms` object is a collection of atoms.  Here
is how to define a N2 molecule by directly specifying the position of
two nitrogen atoms::
"""

from ase import Atoms

d = 1.10
molecule = Atoms('2N', positions=[(0., 0., 0.), (0., 0., d)])
