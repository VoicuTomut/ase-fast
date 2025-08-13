.. module:: ase.calculators.abinit

======
ABINIT
======

ABINIT_ is a density-functional theory code based on pseudopotentials
and a planewave basis.

.. _ABINIT: https://www.abinit.org

Abinit does not specify a default value for the plane-wave cutoff
energy.  You need to set them as in the example at the bottom of the
page, otherwise calculations will fail.  Calculations wihout k-points
are not parallelized by default and will fail! To enable band
paralellization specify ``Number of BanDs in a BLOCK`` (``nbdblock``).

Pseudopotentials
================

Pseudopotentials for ABINIT are available on the
`pseudopotentials`_ website.

.. _pseudopotentials: https://www.abinit.org/pseudopotential.html
