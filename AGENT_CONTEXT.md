# ASE-fast — Agent Context

**Read this before touching any code.**
This is the authoritative reference for LLM agents working on this fork of ASE.
It covers the codebase, the Rust layer, the build system, performance numbers,
and the traps that will waste your time.

---

## What This Project Is

**ASE-fast** is a fork of ASE (Atomic Simulation Environment) with four optional
Rust extensions that accelerate the most expensive hot paths. No user code changes
required — install the extensions and every compatible call is automatically
routed to the fast path.

Package name on PyPI: `ase-fast` (installs the `ase` Python namespace).
Install with Rust: `pip install ase-fast` (auto-builds Rust via `setuptools-rust`).
The Python fallback is always active if Rust is not available.

---

## Rust Extensions — What Exists

Four Rust crates live inside the repo alongside the Python source:

| Crate directory        | Module installed as    | Flag                          | What it accelerates                         |
|------------------------|------------------------|-------------------------------|---------------------------------------------|
| `ase-neighborlist-rs/` | `ase._neighborlist_rs` | `nl._HAVE_RUST_NEIGHBORLIST`  | Scalar, per-atom, dict neighbor lists       |
| `ase-extxyz-rs/`       | `ase._extxyz_rs`       | `extxyz._HAVE_RUST_EXTXYZ`    | extxyz write (atom body) + read (atom lines)|
| `ase-geometry-rs/`     | `ase._geometry_rs`     | `mink._HAVE_RUST_GEOM`        | Minkowski reduction, supercell positions    |
| `ase-io-rs/`           | `ase._io_rs`           | `vasp._HAVE_RUST_IO`          | VASP POSCAR read/write, simple XYZ          |

**Verify all are active:**
```python
import ase.neighborlist as nl
import ase.io.extxyz as extxyz
import ase.geometry.minkowski_reduction as mink
import ase.io.vasp as vasp
assert nl._HAVE_RUST_NEIGHBORLIST, "NL Rust not found"
assert extxyz._HAVE_RUST_EXTXYZ,  "extxyz Rust not found"
assert mink._HAVE_RUST_GEOM,      "geometry Rust not found"
assert vasp._HAVE_RUST_IO,        "IO Rust not found"
```

**The dispatch pattern** (same in every module):
```python
try:
    from ase._neighborlist_rs import primitive_neighbor_list_rs as _pnl_rs, ...
    _HAVE_RUST_NEIGHBORLIST = True
except ImportError:
    _HAVE_RUST_NEIGHBORLIST = False

def primitive_neighbor_list(...):
    if _HAVE_RUST_NEIGHBORLIST and np.isscalar(cutoff):
        return _pnl_rs(...)   # fast path
    return _pnl_python(...)   # fallback
```

---

## Performance Results

Measured on Apple M-series arm64, Python 3.13, `--release` build.
Source: `benchmarks/results/phase11_final.json`.

| Operation                       | Scale          | Python  | ase-fast | Speedup   |
|---------------------------------|----------------|---------|----------|-----------|
| Neighbor list (scalar cutoff)   | 864 atoms      | 7.0 ms  | 0.50 ms  | **14.1x** |
| Neighbor list (scalar cutoff)   | 500 atoms      | 5.9 ms  | 0.35 ms  | **17.0x** |
| Neighbor list (per-atom radii)  | 864 atoms      | 6.6 ms  | 0.62 ms  | **10.7x** |
| Neighbor list (dict cutoffs)    | 864 atoms      | 6.9 ms  | 0.83 ms  |  **8.4x** |
| extxyz write                    | 200fr x 108at  | 31.6 ms | 8.7 ms   |  **3.6x** |
| extxyz read                     | 200fr x 108at  | 1.1 ms  | 1.0 ms   |  **1.1x** |
| VASP POSCAR write               | 2048 atoms     | 2.3 ms  | 0.35 ms  |  **6.7x** |
| VASP POSCAR read                | 2048 atoms     | 1.3 ms  | 0.37 ms  |  **3.5x** |
| Simple XYZ write                | 50fr x 108at   | 5.0 ms  | 1.1 ms   |  **4.5x** |
| Simple XYZ read                 | 50fr x 108at   | 4.3 ms  | 1.7 ms   |  **2.5x** |
| Minkowski reduction             | triclinic cell | 0.07 ms | 0.03 ms  |  **2.6x** |

---

## Build System

### pyproject.toml
```toml
[build-system]
requires = ['setuptools >= 77.0.3', 'setuptools-rust >= 1.9']
build-backend = 'setuptools.build_meta'

[project]
name = 'ase-fast'   # PyPI name — Python namespace is still 'ase'
```

Package discovery uses `include = ['ase*']` to exclude `target/` and `ase-*-rs/`
from being picked up as Python packages.

### setup.py — critical parameters
```python
RustExtension(
    "ase._neighborlist_rs",
    "ase-neighborlist-rs/Cargo.toml",
    binding=Binding.PyO3,
    optional=True,   # no Rust toolchain = Python fallback, not a hard error
    debug=False,     # ALWAYS REQUIRED — see trap #1 below
    features=['pyo3/extension-module'],
)
```

### Cargo workspace (root Cargo.toml)
```toml
[workspace]
members = ["ase-neighborlist-rs", "ase-extxyz-rs", "ase-geometry-rs", "ase-io-rs"]
resolver = "2"

[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```
Member crates must NOT have their own `[profile.release]` — workspace root owns it.

---

## Codebase Map

```
ase/
├── atoms.py          2173 lines  THE central object. Everything accepts Atoms.
├── atom.py            ~300 lines  Single-atom wrapper around Atoms row.
├── cell.py            ~400 lines  Unit cell: 3x3 matrix + PBC logic.
├── symbols.py         ~250 lines  Chemical symbols <-> atomic numbers.
├── units.py            ~80 lines  Physical constants. Import, don't redefine.
├── neighborlist.py   1210 lines  TWO implementations (see below). Rust-accelerated.
│
├── calculators/
│   ├── calculator.py  1181 lines  Base classes. Read before any calculator work.
│   ├── singlepoint.py  ~200 lines  Stores DFT results. Used everywhere in IO.
│   ├── emt.py          ~150 lines  Fast empirical calculator. Good for testing.
│   └── [60+ others]               Each wraps an external code (VASP, GPAW, etc.)
│
├── io/
│   ├── __init__.py    ~300 lines  Public API: read(), write(), iread()
│   ├── formats.py    1000 lines  Format registry. All 86 formats listed here.
│   ├── extxyz.py     1013 lines  THE ML format. Rust-accelerated (write+read).
│   ├── vasp.py       1016 lines  VASP POSCAR/OUTCAR. Rust-accelerated.
│   ├── xyz.py         ~200 lines  Plain XYZ. Rust-accelerated.
│   └── cif.py         901 lines  Crystal structures. Pure Python. Next target.
│
├── build/
│   ├── bulk.py        374 lines  bulk('Cu', 'fcc', a=3.6) -> Atoms
│   ├── surface.py     560 lines  surface('Cu', (1,1,1), 3) -> Atoms
│   ├── supercells.py  435 lines  make_supercell(prim, P) -> Atoms. Rust-accelerated.
│   └── tools.py       535 lines  add_vacuum, sort, rotate, stack, etc.
│
├── geometry/
│   ├── geometry.py    456 lines  find_mic, wrap_positions, get_distances
│   ├── minkowski_reduction.py    Cell reduction. Rust-accelerated.
│   └── analysis.py    663 lines  RDF. Next Rust target.
│
├── optimize/          BFGS, FIRE, L-BFGS, precon variants.
├── md/                VelocityVerlet, Langevin, NVT, NPT.
├── mep/neb.py        1295 lines  Nudged Elastic Band.
└── db/                ASE database (SQLite/PostgreSQL).
```

### Rust crate internals

```
ase-neighborlist-rs/src/
├── lib.rs        PyO3 bindings: primitive_neighbor_list_rs (scalar),
│                 primitive_neighbor_list_radii_rs (per-atom), _dict_rs
└── cell_list.rs  Core cell-list + solve3x3 (Cramer's rule inverse)

ase-extxyz-rs/src/
├── lib.rs        PyO3 bindings: write_atoms_rs, read_atom_lines_rs
├── writer.rs     Atom body serialization (symbols + float arrays)
└── reader.rs     Atom line parsing (typed columns -> numpy arrays)

ase-geometry-rs/src/
└── lib.rs        PyO3 bindings: minkowski_reduce_rs, make_supercell_pos_rs

ase-io-rs/src/
├── lib.rs        PyO3 bindings: parse/format_poscar_positions_rs,
│                 parse/format_xdatcar_config_rs, parse/format_xyz_block_rs
├── vasp.rs       POSCAR position block parser/formatter
└── xyz.rs        Simple XYZ parser/formatter
```

---

## The Two Neighbor List Implementations

**Always know which one you are dealing with.**

### 1. `primitive_neighbor_list()` — function at line 162
Old path. Bin-based cell list, pure NumPy. Called by `neighbor_list()` (line 536).
**Rust ACTIVE.** All three cutoff modes are fast.

### 2. `NewPrimitiveNeighborList` — class at line 780
Newer path. Uses `scipy.cKDTree`. Wrapped by `NeighborList` (line 1106).
**Not Rust-accelerated.** cKDTree is already in C via scipy.

**Quick reference:**
- `neighbor_list('ij', atoms, 3.0)` → path 1 → **Rust ACTIVE**
- `NeighborList([2.5]).update(atoms)` → path 2 → scipy cKDTree

### Fixed bug: solve3x3 transpose (2026-03-28)
The original Rust `solve3x3` in `cell_list.rs` computed `pos @ M^{-T}` instead
of `pos @ M^{-1}` — caused wrong neighbor counts for non-orthogonal cells
(hexagonal, triclinic). Fixed by correcting column index in the final matrix
multiply. If wrong neighbor counts appear for non-cubic cells, check this first.

---

## Calculator Protocol

```python
class Calculator:
    def calculate(self, atoms, properties, system_changes):
        self.results = {
            'energy': float,       # eV
            'forces': np.ndarray,  # (N, 3), eV/Angstrom
            'stress': np.ndarray,  # (6,), Voigt, eV/Angstrom^3
        }
```

- `atoms.get_potential_energy()` -> `atoms.calc.get_property('energy', atoms)`
- `SinglePointCalculator` stores pre-computed DFT results. Used everywhere in IO.

---

## IO System

```python
ase.io.read('myfile.xyz')            # auto-detects from extension
ase.io.read('myfile', format='xyz')  # explicit
```

Format registry in `ase/io/formats.py`. To add a new fast parser, register it
there with the same name — existing dispatch routes through automatically.

`extxyz` vs `xyz`: same `.xyz` extension, distinguished by sniffing comment line.
`extxyz` = extended XYZ with `key=value` pairs. Used for ML datasets.

---

## How to Add a New Rust Extension

### 1. Create the crate

`ase-mymodule-rs/Cargo.toml`:
```toml
[package]
name = "ase-mymodule-rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "_mymodule_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
ndarray = "0.16"
# NO [profile.release] here — workspace root owns it
```

`ase-mymodule-rs/pyproject.toml` (for standalone `maturin develop`):
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"
[project]
name = "ase-mymodule-rs"
requires-python = ">=3.9"
[tool.maturin]
module-name = "ase._mymodule_rs"
features = ["pyo3/extension-module"]
```

### 2. Write the Rust code

`src/lib.rs`:
```rust
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};

#[pyfunction]
fn my_fast_fn<'py>(py: Python<'py>, data: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
    let arr = data.as_array();
    // ... fast Rust code ...
    result.into_pyarray(py).into()
}

#[pymodule]
fn _mymodule_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(my_fast_fn, m)?)?;
    Ok(())
}
```

### 3. Register in workspace and setup.py

Root `Cargo.toml`:
```toml
members = [..., "ase-mymodule-rs"]
```

`setup.py`:
```python
_RUST_EXTENSIONS = [
    ...
    ('ase._mymodule_rs', 'ase-mymodule-rs/Cargo.toml'),
]
```

### 4. Add Python dispatch

```python
try:
    from ase._mymodule_rs import my_fast_fn as _my_fast_fn
    _HAVE_RUST_MYMODULE = True
except ImportError:
    _HAVE_RUST_MYMODULE = False

def my_function(data):
    if _HAVE_RUST_MYMODULE:
        return _my_fast_fn(data)
    return _my_function_python(data)
```

### 5. Build
```bash
pip install -e .    # compiles all crates in release mode, deploys .so files
```

---

## Traps and Pitfalls

### TRAP 1: `debug=False` in RustExtension — CRITICAL
```python
RustExtension("ase._neighborlist_rs", ..., debug=False)  # REQUIRED
```
Without `debug=False`, `pip install -e .` compiles a debug build:
**1.8MB .so, 30x slower** (NL: 0.5ms -> 29ms).
`setuptools-rust` defaults to debug for editable installs. Always set `debug=False`.

### TRAP 2: `[profile.release]` only in workspace root
Member crates with their own `[profile.release]` have it silently ignored.
Only the root `Cargo.toml` profile applies. Delete profiles from member crates.

### TRAP 3: `--quick` benchmarks give wrong speedups
With n=3 reps, sub-millisecond operations are dominated by OS scheduling noise.
Neighbor list shows 0.2x speedup instead of 14x in `--quick` mode.
Use full benchmarks for any published numbers.

### TRAP 4: maturin .so install path
`maturin develop` installs to `site-packages/_neighborlist_rs/` (nested).
The .so must be at `ase/ase/_neighborlist_rs.cpython-*.so` (flat, in ase package).
`pip install -e .` handles this correctly. `maturin develop` alone does NOT.

### TRAP 5: ARM BLAS RuntimeWarnings -> test failures
On Apple Silicon, BLAS emits `RuntimeWarning: divide by zero / overflow` for
NaN/inf inputs in matmul/dot/vecdot. ASE conftest.py converts ALL RuntimeWarnings
to errors. Fix: wrap operations with NaN/inf inputs in:
```python
with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    result = np.dot(a, b)
```
Applied to ~15 source files in 2026-03-28. Add this guard to any new code
that performs matrix operations on potentially-NaN arrays.

### TRAP 6: `atoms.positions` vs `atoms.get_positions()`
NOT equivalent. `get_positions()` respects constraints and can wrap to cell.
`atoms.positions` is the raw numpy array. Do not swap them.

### TRAP 7: PBC handling
Every geometry operation must handle periodic boundary conditions.
`find_mic()` = minimum image convention = correct tool.
Never `np.linalg.norm(pos_i - pos_j)` without a PBC check.

### TRAP 8: `ase/_4/` — do not touch
Compatibility shim for ASE3->ASE4 migration. Ignore it entirely.

---

## Test Infrastructure

All 3480 tests pass (0 failures, 0 errors) as of 2026-03-28.

```bash
pytest ase/test/ -x -q              # full suite (~60s)
pytest ase/test/ -k neighborlist    # neighbor list only
pytest ase/test/ -k vasp            # VASP IO only
pytest ase/test/test_atoms.py       # atoms smoke test
```

---

## Benchmarks

```bash
python benchmarks/run_benchmarks.py                   # full suite, save JSON
python benchmarks/run_benchmarks.py --output out.json # explicit output path
```

Results in `benchmarks/results/`. Authoritative: `benchmarks/results/phase11_final.json`.
The benchmark toggles `_HAVE_RUST_*` flags to get both Python and Rust numbers
in one run without re-importing modules.

---

## What the ML Community Uses (priority order)

These are the APIs that MACE, FAIR-Chem, GNoME users actually call:

1. `ase.io.read('dataset.xyz', index=':')` — read all frames from extxyz
2. `ase.neighborlist.neighbor_list('ijdDS', atoms, cutoff)` — build neighbor list
3. `atoms.get_positions()`, `atoms.get_atomic_numbers()`, `atoms.get_cell()`
4. `atoms.info` and `atoms.arrays` — per-config and per-atom properties
5. `ase.build.bulk('Cu', 'fcc', a=3.6)` — create test structures
6. `ase.io.write('out.xyz', traj, format='extxyz')` — write trajectories

Items 1, 2, 6 are Rust-accelerated.

---

## Remaining Opportunities

| Target                            | File / Lines                      | Gain    | Difficulty |
|-----------------------------------|-----------------------------------|---------|------------|
| extxyz comment parser             | `extxyz.py:67` `key_val_str_to_dict` | 5-10x | Medium     |
| RDF per-frame loop                | `geometry/rdf.py:143`             | 5-15x   | Medium     |
| CIF parser                        | `io/cif.py`                       | 5-10x   | Hard       |
| bothways loop in NewPrimitiveNL   | `neighborlist.py:1033`            | 3-5x    | Easy       |

---

## License

- Modified ASE Python files: **LGPL-2.1** (contribute back upstream)
- New Rust crates (`ase-*-rs/`): **MIT** (max adoption)
- New `ase.ai` LLM layer: **proprietary** (commercial value, Phase 12)
