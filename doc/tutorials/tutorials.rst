.. _old_tutorials:

Tutorials (legacy format)
=========================

Many tutorials have been modernized and moved to :ref:`tutorials`.
This page is going to disappear.

Python
------

If you are not familiar with Python please read :ref:`what is python`.

.. toctree::
   :hidden:

   ../python

If your ASE scripts make extensive use of matrices you may want to familiarize yourself with :ref:`numpy`.

ASE
---

Most of the tutorials will use the :mod:`EMT <ase.calculators.emt>` potential,
but any other :mod:`Calculator <ase.calculators>` could be plugged in instead.

Basic property calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   atomization
   eos/eos

Surface adsorption
^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   db/db

Global optimization
^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   minimahopping/minimahopping

.. note::

  The :mod:`ase.ga` package has moved to
  `ase-ga <https://dtu-energy.github.io/ase-ga/>`_
  including tutorials.

Calculating diffusion/dissociation properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   neb/diffusion
   constraints/diffusion
   dissociation
   neb/idpp
   selfdiffusion/al110

ASE database
^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   tut06_database/database

Molecular Dynamics
^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   tipnp_equil/tipnp_equil

Uncategorized
^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   defects/defects
   qmmm/qmmm
   dimensionality/dimensionality
   deltacodesdft/deltacodesdft
   wannier/wannier
   tut03_vibrations/vibrations


Further reading
---------------

For more details:

* Look at the documentation for the individual :ref:`modules <ase>`.
* Browse the :git:`source code <>` online.
* `External tutorial part of Openscience and ASE workshop, Daresbury 2023  <https://ase-workshop-2023.github.io/tutorial/>`_
