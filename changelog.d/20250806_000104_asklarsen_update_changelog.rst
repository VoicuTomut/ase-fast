.. A new scriv changelog fragment.
..
.. Uncomment the header that is right (remove the leading dots).
..
.. I/O
.. ---
..
.. - A bullet item for the I/O category.
..
.. Calculators
.. -----------
..
.. - A bullet item for the Calculators category.
..
.. Optimizers
.. ----------
..
.. - A bullet item for the Optimizers category.
..
.. Molecular dynamics
.. ------------------
..
.. - A bullet item for the Molecular dynamics category.
..
.. GUI
.. ---
..
.. - A bullet item for the GUI category.
..
.. Development
.. -----------
..
.. - A bullet item for the Development category.
..
.. Documentation
.. -------------
..
.. - A bullet item for the Documentation category.
..
.. Other changes
.. -------------
..
.. - A bullet item for the Other changes category.
..
Bugfixes
--------

- Fixed bug in :class:`io.utils.PlottingVariables` where automatic 
  bounding boxes were incorrectly centered due the image center not being 
  scaled for paper space.

- Fixed bug in :class:`io.pov.POVRAY` where unspecified image (canvas) 
  dimensions would use defaults with an incorrect aspect ratio.
