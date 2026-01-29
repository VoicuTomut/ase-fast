
.. _old_tutorials:

Tutorials (legacy format)
=========================

.. note::

   We are porting the ASE tutorials to sphinx-gallery.
   Tutorials in this section will be ported and moved
   to :ref:`tutorials`.

ASE
---

Most of the tutorials will use the :mod:`EMT <ase.calculators.emt>` potential,
but any other :mod:`Calculator <ase.calculators>` could be plugged in instead.

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
   selfdiffusion/al110


.. original toctree was (in case we want to try to keep the ordering when
.. moving to sphinx-gallery):
..   neb/diffusion
..   constraints/diffusion
..   dissociation
..   neb/idpp  [ported to new tutorials]
..   selfdiffusion/al110

Uncategorized
^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   defects/defects
   qmmm/qmmm
   wannier/wannier
   tut03_vibrations/vibrations


Further reading
---------------

For more details:

* Look at the documentation for the individual :ref:`modules <ase>`.
* Browse the :git:`source code <>` online.
* `External tutorial part of Openscience and ASE workshop, Daresbury 2023  <https://ase-workshop-2023.github.io/tutorial/>`_
