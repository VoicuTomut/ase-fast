.. _changelog:

=========
Changelog
=========

Git master branch
=================

.. CHANGELOG HOWTO.

   To add an entry to the changelog, create a file named
   <timestamp>_<subject>.rst inside the ase/changelog.d/ directory.
   Timestamp should be at least YYYYMMDD.

   You can also install scriv (https://pypi.org/project/scriv/) and run
   "scriv create" to do this automatically, if you do this often.

   Edit the file following a similar style to other changelog entries and
   try to choose an existing section for the release note.

   For example ase/changelog.d/20250108_amber_fix_velocities.rst with contents:

     Calculators
     -----------

     - Amber: Fix scaling of velocities in restart files (:mr:`3427`)

   For each release we generate a full changelog which is inserted below.

.. scriv-auto-changelog-start

Version 3.27.0
==============


Thermochemistry
---------------

- Multiple changes to the :mod:`~ase.thermochemistry` module (:mr:`3358`):

  * **Breaking change**: The ``vib_energies`` property of thermochemistry classes is
    now deprecated. It will still be around for a while until all classes moved to
    the new modes-based framework. Adapt your workflows accordingly. See also below
    for more details on the new modes-based framework.

  * Major parts of the thermochemistry module have been rewritten to include
    a range of new methods: :class:`MSRRHOThermo` based on the modified
    rigid-rotor-harmonic-oscillator (msRRHO) approximation by Grimme *et al.*
    (:doi:`10.1002/chem.201200497` and :doi:`10.1039/D1SC00621E`) and Otlyotov
    and Minenkov :doi:`10.1002/jcc.27129`.
  * A new base class for thermochemistry, :class:`ase.thermochemistry.ThermoBase`,
    has been introduced to facilitate the implementation of new thermochemistry
    methods.
  * Multiple classes are now based on a framework of individual modes
    rather than just a list of vibrational energies. This allows for a
    flexible mixing of different treatments of vibrational modes
    (e.g., Grimme's msRRHO for low-frequency modes and harmonic
    oscillator for high-frequency modes). Each vibrational mode is
    represented by an instance of the
    :class:`ase.thermochemistry.AbstractMode` class or one of its
    subclasses. Multiple modes are then used to build a
    :class:`ase.thermochemistry.BaseThermoChem` instance or one of its
    subclasses. The old way of passing a list of vibrational energies is
    still supported for backwards compatibility, but it is recommended
    to switch to the new modes-based framework. The ``vib_energies``
    property is still available for backwards compatibility, but it is
    recommended to use the ``modes`` property instead, which returns a
    list of mode instances.


I/O
---

- Add support for reading Mulliken charges in
  :meth:`~ase.io.gamess_us.read_gamess_us_out` and
  :meth:`~ase.io.gamess_us.read_gamess_us_punch` (:mr:`3761`)

- Add support to parse general triclinic boxes in
  :meth:`~ase.io.lammpsdata.read_lammps_data`,
  :meth:`~ase.io.lammpsrun.read_lammps_dump_text`, and
  :meth:`~ase.io.lammpsrun.read_lammps_dump_binary` (:mr:`3797`)

- Add support for reading the energy and the dipole moment from "external" in
  :meth:`ase.io.gaussian.read_gausian_out` (:mr:`3801`)

- ASE now follows the EON convention for handling cell geometries.

- Update :meth:`~ase.io.lammpsdata.write_lammps_data` to parse atom types read
  by :meth:`~ase.io.lammpsdata.read_lammps_data` (:mr:`3847`)

- **Breaking** Trajectories and logfiles passed as filenames are not kept
  open during simulations.  Instead, the file is opened, written to
  (generally by appending), then closed again.  This improves IO safety
  and prevents resource leaks in many cases.  It is still possible to
  pass trajectories and logfiles that are already open, and then the
  caller is responsible for closing them.  Doing so may be beneficial
  in fast runs on slow file systems. (:mr:`3899`, :mr:`3930`)

- Updated :func:`~ase.io.lammpsdata.read_lammps_data` and
  :func:`~ase.io.lammpsdata.write_lammps_data` to parse ``Atom Type Labels``
  (:mr:`3916`)

Calculators
-----------

- Add LAMMPSlib features to support arbitrary startup flags (e.g.
  to enable kokkos) and arbitrary initialization callbacks (e.g.
  calling `lammps.mliap.activate_mliappy`)

- The `Vasp` calculator now has a more generic way of handling pseudopotentials. If the user specifies the `pp_version` keyword argument or equivalent `VASP_PP_VERSION` environment variable, the calculator will look for pseudopotentials in the corresponding VASP-provided pseudopotential directory. Simply download the pseudopotential folders provided by VASP and put them in one parent directory, defined by `VASP_PP_PATH`. If `pp_version` is `None` (default), the `Vasp` calculator will only look for `potpaw` (LDA) and `potpaw_PBE` (PBE) to maintain backwards compatability. If `pp_version="64"` (for instance), the calculator will look for pseudopotentials in the `potpaw.64` and `potpaw_PBE.64`, respectively.
- The `potpaw_GGA` (PW91) pseudpotential folder has been removed. This means setting `xc="PW91"` will no longer use the deprecated PW91 pseudopotential folder. It is recommended the PBE pseudopotentials are used for the PW91 functional instead.

- Updated :class:`~ase.calculators.lammpsrun.LAMMPS` to recognize the LAMMPS ``velocity`` command (:mr:`3805`)

Optimizers
----------

- All optimizers (except GPMin) can now work with flat position and gradient arrays.

- Gradients now have the correct sign in the Optimizable API.
  The Optimizable API is still experimental and may change. (:mr:`3908`)

Molecular dynamics
------------------

- The "NPT" thermostat has been renamed to "MelchionnaNPT" for clearer
  comparison with alternatives, and to avoid giving the impression
  that this is a generally recommended default choice for NPT.

  For backward compatibility an :class:`ase.md.npt.NPT` class remains
  in place, which aliases the renamed class at its new location
  :class:`ase.md.melchionna.MelchionnaNPT`.

  The alias is marked as "deprecated" and will be removed in a future
  ASE version.

- New LangevinBAOAB variable-cell Langevin integrator with Leimkhler's BAOAB method

GUI
---

- Added a history feature, i.e. undo/redo, to the ASE GUI.

Documentation
-------------

 - Move database introduction tutorial to sphinx gallery

 - Move Surface adsorption study using the ASE database  tutorial to sphinx gallery

- Move Move "Calculating Delta-values tutorial" to examples/deltacodesdft sphinx gallery

Other changes
-------------

- Fixed backward-compatibility of
  :class:`~ase.constraints.FixedLine` and :class:`~ase.constraints.FixedPlane`
  made before ASE 3.23.0 to be read with
  :meth:`~ase.constraints.dict2constraint` after ASE 3.23.0 (:mr:`3786`)

* ASE now requires Python 3.10+.

- Fix deprecated SQL syntax :mr:`3815`.

- Added :meth:`ase.lattice.match_to_lattice` which matches an input cell
  to a specific Bravais lattice and returns a list of matching representations.

- The :mod:`ase.ga` module has been moved to the standalone
   `ase-ga <https://dtu-energy.github.io/ase-ga/>`__ project.

- Improved efficiency of :func:`~ase.geometry.rdf.get_rdf` using
  :class:`~ase.neighborlist.NeighborList` (:mr:`3888`)

Bugfixes
--------

- Remove subscripting of images iterable in extxyz writer so it works
  with non-subscriptable iterables when "move_mask" key is present in
  `atoms.arrays` (:mr:`3795`)

- Fix :func:`~ase.io.extxyz.write_extxyz` to ignore constraints other than
  :class:`~ase.constraints.FixAtoms` and :class:`~ase.constraints.FixCartesian`
  when ``move_mask`` is present in ``columns`` (default since ASE 3.26.0)
  (:mr:`3808`)

- Fix :func:`~ase.io.extxyz.write_extxyz` to handle multiple
  :class:`~ase.constraints.FixAtoms` correctly (:mr:`3812`)

- Fix :func:`~ase.io.extxyz.write_extxyz` to handle unconstrained atoms in
  :class:`~ase.Atoms` with :class:`~ase.constraints.FixCartesian` correctly
  (:mr:`3812`)

- Make :class:`~ase.calculators.vasp.Vasp` calculator INCAR reader use first instance of each tag,
  to be same as VASP code itself (:mr:`3842`)

- Fix constraints from first frame being applied to all frames (leading to errors) in :func:`~ase.io.extxyz.write_extxyz` (:mr:`3920`)

- Fixed partial RDFs in :func:`~ase.geometry.rdf.get_rdf` to be conssitent with
  the common definition in, e.g., LAMMPS ``compute rdf`` (:mr:`3921`)

- Fixed :class:`~ase.calculators.lammpsrun.LAMMPS` to be robust against LAMMPS log files with non-ASCII characters (:mr:`3925`)

- Fixed :class:`~ase.calculators.lammpsrun.LAMMPS` to not run the default `fix nve` command when users set `fix` explicitly (:mr:`3805`)

Version 3.26.0
==============

I/O
---

- Added communicator argument to parprint, which defaults to world if None, analogous as for paropen

- Added single float encoding for :mod:`~ase.io.jsonio` (:mr:`3682`)

- Changed :func:`~ase.io.extxyz.write_extxyz` to store
  :class:`~ase.constraints.FixAtoms` and
  :class:`~ase.constraints.FixCartesian` by default without explicitly
  specifying ``move_mask`` in ``columns`` (:mr:`3713`)

- **Breaking change**: Removed IOFormat.open() method. It is untested and appears to be unused. :mr:`3738`

- Fix :func:`~ase.io.vasp.read_vasp` to correctly read both atomic and lattice velocities if present in POSCAR (:mr:`3762`)

Calculators
-----------

- Added per-atom ``energies`` consistent with LAMMPS to
  :class:`~ase.calculators.tersoff.Tersoff` (:mr:`3656`)

- Added toggles between analytical and numerical forces/stress in
  :class:`~ase.calculators.fd.FiniteDifferenceCalculator` (:mr:`3678`)

- Added calculators ``mattersim`` and ``mace_mp`` to the ``get_calculator()`` function

- Changed :class:`~ase.calculators.elk.ELK` based on
  :class:`~ase.calculators.GenericFileIOCalculator` (:mr:`3736`)

- DFTD3 no longer warns about systems that are neither 3D periodic
  nor 0D, because there is no way to adapt the code that resolves the
  condition warned about.  (:mr:`3740`)

Optimizers
----------

- Logfile and trajectory inputs now accept both string and Path objects.

- **Breaking change:** The :class:`~ase.utils.abc.Optimizable` interface
  now works in terms of arbitrary degrees of freedom rather than
  Cartesian (Nx3) ones.
  Please note that the interface is still considered an internal feature
  and may still change significantly. (:mr:`3732`)

Molecular dynamics
------------------

- Added anisotropic NpT with MTK equations (:mr:`3595`).

- Fixed bug in Nose-Hoover chain thermostat which would inconsistently update extended variables for the thermostat.

GUI
---

- Atomic spins can now be visualized as arrows

- Mouse button 2 and 3 are now equivalent in the GUI, which simplifies
  life on particularly MacOS (:mr:`3669`).
- Menu shortcut keys now work as expected on MacOS.
- In Rotate and Translate mode, Ctrl + arrow key now works as intended on
  MacOS.  Left alt and Command now have the same effect (:mr:`3669`).

- Changed Alt+X, Alt+Y, Alt+Z to Shift+X, Shift+Y, Shift+Z to view planes from "other side"
- Changed views into basis vector planes to I, J, K, Shift+I, Shift+J, Shift+K

- Added general window to view and edit data on atoms directly
  in the same style as the cell editor.
  The window currently edits
  symbols and Cartesian positions only (:mr:`3790`).

Development
-----------

- Enable ruff on all scripts inside documentation

Documentation
-------------

- Web page now uses sphinx book theme (:mr:`3684`).

- Documentation moved to `ase-lib.org <https://ase-lib.org/>`_.

Other changes
-------------

- Removed ``Quaternions`` (subclass of ``Atoms``).
  The ``quaternions`` read from a LAMMPS data file is still accessible as an array
  in ``Atoms``. (:mr:`3709`)

- Re-added the ``spin`` option of
  :meth:`~ase.spectrum.band_structure.BandStructurePlot.plot`
  to plot only the specified spin channel (:mr:`3726`)

Bugfixes
--------

- Fixed :class:`~ase.calculators.tersoff.Tersoff` to compute properties
  correctly (:mr:`3653`, :mr:`3655`, :mr:`3657`).

- Enable :func:`ase.io.magres.read_magres` to handle cases from CASTEP < 23 where indices and labels are "munged" together if the index exceeds 99. If an index exceeds 999 the situation remains ambiguous and an error will be raised. (:mr:`3530`)

- Fix duplicated transformation (e.g. rotation) of symmetry labels in :func:`~ase.dft.bz.bz_plot` (:mr:`3617`).

- Fixed bug in :class:`io.utils.PlottingVariables` where automatic
  bounding boxes were incorrectly centered due the image center not being
  scaled for paper space (:mr:`3769`).

- Fixed bug in :class:`io.pov.POVRAY` where unspecified image (canvas)
  dimensions would use defaults with an incorrect aspect ratio (:mr:`3769`).

Structure tools
---------------

- Added ``score_key='metric'`` to :func:`~ase.build.find_optimal_cell_shape`
  for scoring a cell based on its metric tensor (:mr:`3616`)

Version 3.25.0
==============

I/O
---

- Moved Postgres, MariaDB and MySQL backends to separate project:
  https://gitlab.com/ase/ase-db-backends.  Install from PyPI with
  ``pip install ase-db-backends`` (:mr:`3545`).

- **BREAKING** ase.io.orca ``read_orca_output`` now returns Atoms with attached properties.
  ``ase.io.read`` will use this function.
  The previous behaviour (return results dictionary only) is still available from function ``read_orca_outputs``. (:mr:`3599`)

- Added :func:`~ase.io.castep.write_castep_geom` and
  :func:`~ase.io.castep.write_castep_md` (:mr:`3229`)

- Fixed :mod:`ase.data.pubchem` module to convert ``#`` in SMILES to HEX
  ``%23`` for URL (:mr:`3620`).

- :mod:`ase.db`: Unique IDs are now based on UUID rather than pseudorandom numbers that could become equal due to seeding (:mr:`3614`).
- :mod:`ase.db`: Fix bug where unique_id could be converted to float or int (:mr:`3613`).
- Vasp: More robust reading of CHGCAR (:mr:`3607`).
- Lammpsdump: Read timestep from lammpsdump and set element based on mass (:mr:`3529`).
- Vasp: Read and write velocities (:mr:`3597`).
- DB: Support for LMDB via `ase-db-backends` project (:mr:`3564`, :mr:`3639`).
- Espresso: Fix bug reading `alat` in some cases (:mr:`3562`).
- GPAW: Fix reading of total charge from text file (:mr:`3519`).
- extxyz: Somewhat restrict what properties are automatically written (:mr:`3516`).
- Lammpsdump: Read custom property/atom LAMMPS dump data (:mr:`3510`).

Calculators
-----------

- More robust reading of Castep XC functional (:mr:`3612`).
- More robust saving of calculators to e.g. trajectories (:mr:`3610`).
- Lammpslib: Fix outdated MPI check (:mr:`3594`).
- Morse: Optionally override neighbor list implementation (:mr:`3593`).
- EAM: Calculate stress (:mr:`3581`).

- A new Calculator :class:`ase.calculators.tersoff.Tersoff` has been added. This is a Python implementation of a LAMMPS-style Tersoff interatomic potential. Parameters may be passed directly to the calculator as a :class:`ase.calculators.tersoff.TersoffParameters` object, or the Calculator may be constructed from a LAMMPS-style file using its ``from_lammps`` classmethod. (:mr:`3502`)

Optimizers
----------

- Fix step counting in the
  :class:`~ase.optimize.cellawarebfgs.CellAwareBFGS` (:mr:`3588`).

- Slightly more efficient/robust GoodOldQuasiNewton (:mr:`3570`).

Molecular dynamics
------------------

- Merged ``self.communicator`` into ``self.comm`` (:mr:`3631`).

- Improved random sampling in countour exploration (:mr:`3643`).
- Fix small energy error in Langevin dynamics (:mr:`3567`).
- Isotropic NPT with MTK equations (:mr:`3550`).
- Bussi dynamics now work in parallel (:mr:`3569`).
- Improvements to documentation (:mr:`3566`).
- Make Nose-Hoover chain NVT faster and fix domain decomposition
  with Asap3 (:mr:`3571`).

- NPT now works with cells that are upper or lower triangular matrices
  (:mr:`3277`) aside from upper-only as before.

- Fix inconsistent :meth:`irun` for NPT (:mr:`3598`).

GUI
---

- Fix windowing bug on WSL (:mr:`3478`).

- Added button to wrap atoms into cell (:mr:`3587`).

Development
-----------

- Changelog is now generated using ``scriv`` (:mr:`3572`).

- CI cleanup; pypi dependencies in CI jobs are now cached
  (:mr:`3628`, :mr:`3629`).
- Maximum automatic pytest workers reduced to 8 (:mr:`3628`).

- Ruff formatter to be gradually enabled across codebase (:mr:`3600`).

Other changes
-------------

- :meth:`~ase.cell.Cell.standard_form` can convert to upper triangular (:mr:`3623`).
- Bugfix: :func:`~ase.geometry.geometry.get_duplicate_atoms` now respects pbc (:mr:`3609`).
- Bugfix: Constraint masks in cell filters are now respected down to numerical precision.  Previously, the constraints could be violated by a small amount (:mr:`3603`).
- Deprecate :func:`~ase.utils.lazyproperty` and :func:`~ase.utils.lazymethod`
  since Python now provides :func:`functools.cached_property` (:mr:`3565`).
- Remove ``nomad-upload`` and ``nomad-get`` commands due to incompatibility
  with recent Nomad (:mr:`3563`).
- Fix normalization of phonon DOS (:mr:`3472`).
- :class:`~ase.io.utils.PlottingVariables` towards rotating the
  camera rather than the atoms (:mr:`2895`).

.. scriv-auto-changelog-end


Version 3.24.0
==============

Requirements
------------

* The minimum supported Python version has increased to 3.9 (:mr:`3473`)
* Support numpy 2 (:mr:`3398`, :mr:`3400`, :mr:`3402`)
* Support spglib 2.5.0 (:mr:`3452`)

Atoms
-----
* New method :func:`~ase.Atoms.get_number_of_degrees_of_freedom()` (:mr:`3380`)
* New methods :func:`~ase.Atoms.get_kinetic_stress()`, :func:`~ase.Atoms.get_kinetic_stresses()` (:mr:`3362`)
* Prevent truncation when printing Atoms objects with 1000 or more atoms (:mr:`2518`)

DB
--
* Ensure correct float format when writing to Postgres database (:mr:`3475`)

Structure tools
---------------

* Add atom tagging to ``ase.build.general_surface`` (:mr:`2773`)
* Fix bug where code could return the wrong lattice when trying to fix the handedness of a 2D lattice  (:mr:`3387`)
* Major improvements to :func:`~ase.build.find_optimal_cell_shape`: improve target metric; ensure rotationally invariant results; avoid negative determinants; improved performance via vectorisation (:mr:`3404`, :mr:`3441`, :mr:`3474`). The ``norm`` argument to :func:`~ase.build.supercells.get_deviation_from_optimal_cell_shape` is now deprecated.
* Performance improvements to :class:`ase.spacegroup.spacegroup.Spacegroup` (:mr:`3434`, :mr:`3439`, :mr:`3448`)
* Deprecated :func:`ase.spacegroup.spacegroup.get_spacegroup` as results can be misleading (:mr:`3455`).
  

Calculators / IO
----------------

* Amber: Fix scaling of velocities in restart files (:mr:`3427`)
* Amber: Raise an error if cell is orthorhombic (:mr:`3443`)
* CASTEP

  - **BREAKING** Removed legacy ``read_cell`` and ``write_cell`` functions from ase.io.castep. (:mr:`3435`)
  - .castep file reader bugfix for Windows (:mr:`3379`), testing improved (:mr:`3375`)
  - fix read from Castep geometry optimisation with stress only (:mr:`3445`)

* EAM: Fix calculations with self.form = "eam" (:mr:`3399`)
* FHI-aims
  
  - make free_energy the default energy (:mr:`3406`)
  - add legacy DFPT parser hook (:mr:`3495`)

* FileIOSocketClientLauncher: Fix an unintended API change (:mr:`3453`)
* FiniteDifferenceCalculator: added new calculator which wraps other calculator for finite-difference forces and strains (:mr:`3509`)
* GenericFileIOCalculator fix interaction with SocketIO (:mr:`3381`)
* LAMMPS

  - fixed a bug reading dump file with only one atom (:mr:`3423`)
  - support initial charges (:mr:`2846`, :mr:`3431`)

* MixingCalculator: remove requirement that mixed calculators have common ``implemented_properties`` (:mr:`3480`)
* MOPAC: Improve version-number parsing (:mr:`3483`)
* MorsePotential: Add stress (:mr:`3485`)
* NWChem: fixed reading files from other directories (:mr:`3418`)
* Octopus: Improved IO testing (:mr:`3465`)
* ONETEP calculator: allow ``pseudo_path`` to be set in config (:mr:`3385`)
* Orca: Only parse dipoles if COM is found. (:mr:`3426`)
* Quantum Espresso

  - allow arbitrary k-point lists (:mr:`3339`)
  - support keys from EPW (:mr:`3421`)
  - Fix path handling when running remote calculations from Windows (:mr:`3464`)

* Siesta: support version 5.0 (:mr:`3464`)
* Turbomole: fixed formatting of "density convergence" parameter (:mr:`3412`)
* VASP

  - Fixed a bug handling the ICHAIN tag from VTST (:mr:`3415`)
  - Fixed bugs in CHG file writing (:mr:`3428`) and CHGCAR reading (:mr:`3447`)
  - Fix parsing POSCAR scale-factor line that includes a comment (:mr:`3487`)
  - Support use of unknown INCAR keys (:mr:`3488`)
  - Drop "INCAR created by Atomic Simulation Environment" header (:mr:`3488`)
  - Drop 1-space indentation of INCAR file (:mr:`3488`)
  - Use attached atoms if no atom argument provided to :func:`ase.calculators.vasp.Vasp.calculate` (:mr:`3491`)

GUI
---
* Refactoring of :class:`ase.gui.view.View` to improve API for external projects (:mr:`3419`)
* Force lines to appear black (:mr:`3459`)
* Fix missing Alt+X/Y/Z/1/2/3 shortcuts to set view direction (:mr:`3482`)
* Fix incorrect frame number after using Page-Up/Page-Down controls (:mr:`3481`)
* Fix incorrect double application of ``repeat`` to ``energy`` in GUI (:mr:`3492`)

Molecular Dynamics
------------------

* Added Bussi thermostat :class:`ase.md.bussi.Bussi` (:mr:`3350`)
* Added Nose-Hoover chain NVT thermostat :class:`ase.md.nose_hoover_chain.NoseHooverChainNVT` (:mr:`3508`)
* Improve ``force_temperature`` to work with constraints (:mr:`3393`)
* Add ``**kwargs`` to MolecularDynamics, passed to parent Dynamics (:mr:`3403`)
* Support modern Numpy PRNGs in Andersen thermostat (:mr:`3454`)

Optimizers
----------
* **BREAKING** The ``master`` parameter to each Optimizer is now passed via ``**kwargs`` and so becomes keyword-only. (:mr:`3424`)
* Pass ``comm`` to BFGS and CellAwareBFGS as a step towards cleaner parallelism (:mr:`3397`)
* **BREAKING** Removed deprecated ``force_consistent`` option from Optimizer (:mr:`3424`)

Phonons
-------

* Fix scaling of phonon amplitudes (:mr:`3438`)
* Implement atom-projected PDOS, deprecate :func:`ase.phonons.Phonons.dos` in favour of :func:`ase.phonons.Phonons.get_dos` (:mr:`3460`)
* Suppress warnings about imaginary frequencies unless :func:`ase.phonons.Phonons.get_dos` is called with new parameter ``verbose=True`` (:mr:`3461`)

Pourbaix (:mr:`3280`)
---------------------

* New module :mod:`ase.pourbaix` written to replace :class:`ase.phasediagram.Pourbaix`
* Improved energy definition and diagram generation method
* Improved visualisation

Spectrum
--------
* **BREAKING** :class:`ase.spectrum.band_structure.BandStructurePlot`: the ``plot_with_colors()`` has been removed and its features merged into the ``plot()`` method.

Misc
----
* Cleaner bandgap description from :class:`ase.dft.bandgap.GapInfo` (:mr:`3451`)

Documentation
-------------
* The "legacy functionality" section has been removed (:mr:`3386`)
* Other minor improvements and additions (:mr:`2520`, :mr:`3377`, :mr:`3388`, :mr:`3389`, :mr:`3394`, :mr:`3395`, :mr:`3407`, :mr:`3413`, :mr:`3416`, :mr:`3446`, :mr:`3458`, :mr:`3468`)

Testing
-------
* Remove some dangling open files (:mr:`3384`)
* Ensure all test modules are properly packaged (:mr:`3489`)

Units
-----
* Added 2022 CODATA values (:mr:`3450`)
* Fixed value of vacuum magnetic permeability ``_mu0`` in (non-default) CODATA 2018 (:mr:`3486`)

Maintenance and dev-ops
-----------------------
* Set up ruff linter (:mr:`3392`, :mr:`3420`)
* Further linting (:mr:`3396`, :mr:`3425`, :mr:`3430`, :mr:`3433`, :mr:`3469`, :mr:`3520`)
* Refactoring of ``ase.build.bulk`` (:mr:`3390`), ``ase.spacegroup.spacegroup`` (:mr:`3429`)

Earlier releases
================

Releases earlier than ASE 3.24.0 do not have separate release notes and changelog.
Their changes are only listed in the :ref:`releasenotes`.
