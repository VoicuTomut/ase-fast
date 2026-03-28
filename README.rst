ASE-fast — Atomic Simulation Environment (accelerated)
=======================================================

.. image:: https://gitlab.com/ase/ase/-/raw/master/doc/static/ase256.png
    :target: https://ase-lib.org/
    :align: center

|

.. image:: https://badge.fury.io/py/ase.svg
    :target: https://pypi.org/project/ase/

.. image:: https://gitlab.com/ase/ase/badges/master/coverage.svg?job=coverage-combine
    :target: https://ase-lib.org/coverage-html/


**ASE-fast** is a drop-in replacement for ASE with optional Rust-accelerated
hot paths.  No code changes required — install the Rust extensions and every
compatible call is automatically routed to the fast path.

Measured speedups on Apple M-series (arm64, Python 3.13):

+----------------------------------+------------------+
| Operation                        | Speedup          |
+==================================+==================+
| Neighbor list (scalar cutoff)    | **13.6×**        |
+----------------------------------+------------------+
| Neighbor list (per-atom radii)   | **12.2×**        |
+----------------------------------+------------------+
| Neighbor list (dict cutoffs)     | **9.1×**         |
+----------------------------------+------------------+
| VASP POSCAR write                | **6.3×**         |
+----------------------------------+------------------+
| VASP POSCAR read                 | **3.3×**         |
+----------------------------------+------------------+
| Extended XYZ write (large atoms) | **3.5×**         |
+----------------------------------+------------------+
| Simple XYZ write                 | **2.3×**         |
+----------------------------------+------------------+
| Minkowski reduction              | **2.6×**         |
+----------------------------------+------------------+

ASE is a set of tools and Python modules for setting up, manipulating,
running, visualizing and analyzing atomistic simulations.

Upstream webpage: https://ase-lib.org/


Requirements
------------

* Python_ 3.10 or later
* NumPy_ (base N-dimensional array package)
* SciPy_ (library for scientific computing)
* Matplotlib_ (2D Plotting)

Optional:

* Flask_ (for ase.db web-interface)
* spglib_ (for symmetry operations)
* Rust_ + maturin_ (to build the optional Rust fast-path extensions)

Installation
------------

Pure-Python install (same as upstream ASE):

::

  pip install ase

Development install from this repository:

::

  pip install -e .

With Rust acceleration (recommended for MD / ML workflows):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a Rust toolchain (``rustup``) and ``maturin``:

::

  pip install maturin
  # Build and install all four Rust extensions:
  cd ase-neighborlist-rs && maturin develop --release && cd ..
  cd ase-extxyz-rs       && maturin develop --release && cd ..
  cd ase-geometry-rs     && maturin develop --release && cd ..
  cd ase-io-rs           && maturin develop --release && cd ..

That's it.  All Rust crates live inside the repository alongside the Python
source.  After building, ASE automatically detects the extensions and enables
the fast paths — no configuration needed.

To verify the extensions are active:

::

  python -c "
  import ase.neighborlist as nl
  import ase.io.extxyz as ex
  import ase.geometry.minkowski_reduction as mk
  import ase.io.vasp as vasp
  print('NL Rust:', nl._HAVE_RUST_NEIGHBORLIST)
  print('extxyz Rust:', ex._HAVE_RUST_EXTXYZ)
  print('geometry Rust:', mk._HAVE_RUST_GEOM)
  print('VASP IO Rust:', vasp._HAVE_RUST_IO)
  "

Running benchmarks:

::

  python benchmarks/run_benchmarks.py                   # full suite
  python benchmarks/run_benchmarks.py --quick           # fewer reps
  python benchmarks/run_benchmarks.py --output out.json # save results

Testing
-------

Please run the tests::

    $ ase test  # takes 1 min.

and send us the output if there are failing tests.


Contact
-------

* Mailing list: ase-users_

* Chat: Join the ``#ase`` channel on Matrix_, also accessible via the Element_ webclient.

* There is an `ASE forum <https://matsci.org/c/ase/36>`_ on
  the `Materials Science Community Forum <https://matsci.org/>`_.

Feel free to create Merge Requests and Issues on our GitLab page:
https://gitlab.com/ase/ase

For regular support, please use the mailing list or chat rather than GitLab.


Example
-------

Geometry optimization of hydrogen molecule with NWChem:

>>> from ase import Atoms
>>> from ase.optimize import BFGS
>>> from ase.calculators.nwchem import NWChem
>>> from ase.io import write
>>> h2 = Atoms('H2',
               positions=[[0, 0, 0],
                          [0, 0, 0.7]])
>>> h2.calc = NWChem(xc='PBE')
>>> opt = BFGS(h2, trajectory='h2.traj')
>>> opt.run(fmax=0.02)
BFGS:   0  19:10:49    -31.435229     2.2691
BFGS:   1  19:10:50    -31.490773     0.3740
BFGS:   2  19:10:50    -31.492791     0.0630
BFGS:   3  19:10:51    -31.492848     0.0023
>>> write('H2.xyz', h2)
>>> h2.get_potential_energy()  # ASE's units are eV and Ang
-31.492847800329216

This example requires NWChem to be installed.

::

    $ ase gui h2.traj


Contributors
------------

* Original ASE team (DTU Physics and many others) — see upstream AUTHORS file
* **Andrei Voicu Tomut** — Rust acceleration layer (ase-fast), type hints, error messages, LLM builder


.. _Python: https://www.python.org/
.. _NumPy: https://numpy.org/doc/stable/
.. _SciPy: https://docs.scipy.org/doc/scipy/
.. _Matplotlib: https://matplotlib.org/
.. _flask: https://pypi.org/project/Flask/
.. _spglib: https://github.com/spglib/spglib
.. _ase-users: https://listserv.fysik.dtu.dk/mailman/listinfo/ase-users
.. _Matrix: https://matrix.to/#/!JEiuNJLuxedbohAOuH:matrix.org
.. _Element: https://app.element.io/#/room/#ase:matrix.org
.. _Rust: https://www.rust-lang.org/tools/install
.. _maturin: https://www.maturin.rs/
