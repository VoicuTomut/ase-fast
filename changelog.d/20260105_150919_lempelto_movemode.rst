.. A new scriv changelog fragment.
..
.. Uncomment the section that is right (remove the leading dots).
.. For top level release notes, leave all the headers commented out.
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
GUI
---

- Move and Rotate modes no longer turn on if no atoms are selected.

- An indicator is added to the bottom right corner that shows that a movement mode is active. This indicator also includes a contextual hint about modifier keys.

- Rotation vectors are transformed to give a more intuitive pitch/yaw/roll experience in rotate mode which is also consistent with mouse movements.

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

- Ctrl/Shift key handling is refactored in the GUI to try and combat a bug where Num Lock would continuously register as a modifier. This was causing the arrow keys to behave incorrectly in "Move" and "Rotate" modes. 

