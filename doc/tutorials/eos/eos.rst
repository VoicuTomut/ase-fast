.. _eos:

=======================
Equation of state (EOS)
=======================

.. note::

  We are currently moving to a new way to display our examples.
  For this example we have an updated version, which you 
  can find :ref:`here <eos_example>`.
  The example on this page is deprecated and will be removed 
  once all examples have been moved to 
  the new format.


First, do a bulk calculation for different lattice constants:

.. literalinclude:: eos1.py

This will write a trajectory file containing five configurations of
FCC silver for five different lattice constants.  Now, analyse the
result with the :class:`~ase.eos.EquationOfState` class and this
script:

.. literalinclude:: eos2.py

|eos|

A quicker way to do this analysis, is to use the :mod:`ase.gui` tool:

.. highlight:: bash

::

    $ ase gui Ag.traj

And then choose :menuselection:`Tools --> Bulk modulus`.

.. |eos| image:: Ag-eos.png
