.. A new scriv changelog fragment.
..
.. Uncomment the section that is right (remove the leading dots).
.. For top level release notes, leave all the headers commented out.
..
I/O
---

- **Breaking** Trajectories and logfiles passed as filenames are not kept 
  open during simulations.  Instead, the file is opened, written to
  (generally by appending), then closed again.  This improves IO safety
  and prevents resource leaks in many cases.  It is still possible to
  pass trajectories and logfiles that are already open, and then the
  caller is responsible for closing them.  Doing so may be beneficial
  in fast runs on slow file systems. (:mr:`3899`)

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
.. --------
..
.. - A bullet item for the Bugfixes category.
..
