""".. _nanoparticle:

Nanoparticle
==================
This tutorial shows how to use the :mod:`ase.cluster` module
to set up metal nanoparticles with common crystal forms.
Please have a quick look at the documentation.
"""

# %%
# Build and optimise nanoparticle
# -------------------------------
#
# Consider :func:`ase.cluster.Octahedron`.  Aside from generating
# strictly octahedral nanoparticles, it also offers a ``cutoff``
# keyword to cut the corners of the
# octahedron.  This produces "truncated octahedra", a well-known structural
# motif in nanoparticles.  Also, the lattice will be consistent with the bulk
# FCC structure of silver.
#
# .. admonition:: Exercise
#
#    Play around with :func:`ase.cluster.Octahedron` to produce truncated
#    octahedra.  Set up a cuboctahedral
#    silver nanoparticle with 55 atoms.  As always, verify with the ASE GUI that
#    it is beautiful.
#
# ASE provides a forcefield code based on effective medium theory,
# :class:`ase.calculators.emt.EMT`, which works for the FCC metals (Cu, Ag, Au,
# Pt, and friends).  This is much faster than DFT so let's use it to
# optimise our cuboctahedron.
#
# .. admonition:: Exercise
#
#    Optimise the structure of our :mol:`Ag_55` cuboctahedron
#    using the :class:`ase.calculators.emt.EMT`
#    calculator.
#
