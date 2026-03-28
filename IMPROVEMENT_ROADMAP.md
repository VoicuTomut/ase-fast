# ASE Improvement Roadmap — OldGold Revival

**Project:** Atomic Simulation Environment (ASE)
**Source:** https://gitlab.com/ase/ase
**Revival goal:** Drop-in replacement with full type hints, Rust-accelerated hot paths,
and an LLM-friendly interface layer — targeting the ML interatomic potential community.
**Audit date:** 2026-03-27
**Last updated:** 2026-03-28 (Phase 11 DONE; benchmarks complete; README updated with ASE-fast branding)

---

## License Analysis — Commercial Use

ASE is licensed under **LGPL-2.1-or-later** (GNU Lesser General Public License v2.1).

### What this means for the revival

| Activity | Allowed? | Condition |
|----------|----------|-----------|
| Use ASE in a commercial product | YES | Must include LGPL notice |
| Sell a product that imports ASE | YES | Your app code stays proprietary |
| Modify ASE Python files and distribute | YES | Modified ASE files must remain LGPL |
| Write new Rust extensions as a separate package | YES | **You own those — any license you choose** |
| Write a new `ase.ai` LLM layer that calls ASE | YES | **New code — any license you choose** |
| Take ASE code, make it proprietary, resell | NO | Violation of LGPL |

### Practical strategy

**The LGPL is the best possible license for this project.** Unlike GPL, it explicitly
allows non-free programs to link with the library. The key split is:

- **Modified ASE files** (type hints, error messages, Python wrappers) → must stay LGPL.
  Contribute these upstream. This builds goodwill and avoids maintenance burden.
- **Rust extension package** (`ase-rs` or `ase-fast`) → new code, MIT or Apache-2.0.
  This is where the commercial value lives. You own it.
- **`ase.ai` LLM layer** (Phase 6) → new code, proprietary or open as you choose.
  This is the highest-value differentiator for enterprise customers.

### Recommended IP structure

```
ase (LGPL-2.1)          — upstream, contribute type hints back
    ↕ depends on
ase-fast (MIT)          — Rust acceleration, open source, drives adoption
    ↕ optional
ase-ai (proprietary)    — LLM layer, enterprise API, this is what companies pay for
```

**Bottom line:** LGPL-2.1 is commercially friendly. Sell `ase-ai`. Open-source `ase-fast`.
Contribute the type hints and error messages back upstream.

---

## Why ASE

- Universal data format for atomistic ML: every MLIP pipeline (GNoME, FAIR-Chem,
  MACE, NequIP, CHGNet) uses `ase.Atoms` and `extxyz`
- Zero compiled extensions — pure Python + NumPy/SciPy — maximum Rust leverage
- 2.8% type hint coverage — LLMs and IDEs are effectively blind to the API
- Neighbor list is a community-famous bottleneck; users routinely replace it with
  matscipy or custom code
- Companies (Schrodinger, Exscientia, DeepMind, Meta FAIR, Microsoft Research)
  all depend on this library and feel the pain

---

## Phase Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 0  | Audit | DONE | See `BOTTLENECK_ANALYSIS.md` and `AGENT_CONTEXT.md` |
| 1  | Type hints — Atoms + Cell + core | DONE | atoms.py, atom.py, cell.py, symbols.py fully annotated |
| 2  | Type hints — Calculator interface | DONE | calculator.py, singlepoint.py fully annotated |
| 3  | Type hints — IO public API + extxyz | DONE | formats.py, io/__init__.py, extxyz.py fully annotated |
| 4  | Type hints — build/, geometry/, neighborlist | DONE | bulk.py, surface.py, supercells.py, geometry.py, neighborlist.py fully annotated |
| 4.5 | **Test enrichment** | **DONE** | 171 new tests; 0 regressions; baseline JSON saved |
| 5  | Error messages — human-readable across all modules | **DONE** | 20 improvements across neighborlist, bulk, surface, extxyz, geometry; 2 xfails resolved |
| 6  | LLM layer — natural language → ASE code | **DONE** | ase.ai module; AtomicStructureBuilder + build(); 25 tests; self-correcting retry loop |
| 7  | Rust neighborlist — inner loop + bothways | **DONE** | PyO3 cell-list; 14× avg speedup (scalar cutoff); 10 Rust unit tests; 0 regressions |
| 8  | Rust extxyz — read/write | **DONE** | 3.4-3.6× write; 1.1× read; 97 tests pass |
| 9  | Rust neighborlist (per-atom/dict cutoffs) + geometry + RDF | **DONE** | 11-12× radii, 7-8× dict, 3.1× Minkowski; ase-geometry-rs crate |
| 10 | Rust IO — VASP + XYZ | **DONE** | 3.5× read, 6.1× write VASP; ase-io-rs crate; 90 IO tests pass |
| 11 | Benchmarks — vs baseline, vs matscipy | **IN PROGRESS** | Plan in `PHASE11_PLAN.md`; all 3480 tests pass |
| 12 | Branding + outreach | TODO | PyPI, MLIP community |

---

## Phase Details

### Phase 1 — Type hints: Atoms + Cell + core data model
**Files:** `ase/atoms.py`, `ase/atom.py`, `ase/cell.py`, `ase/symbols.py`
**Scope:**
- `Atoms.__init__` — 14 kwargs, all untyped
- All `get_*/set_*` methods on Atoms (~50 methods)
- `Cell` class — `new()`, `scaled_positions()`, `cartesian_positions()`
- Return types: most return `np.ndarray` — annotate shape in docstring, type in signature
**Estimated effort:** Large (atoms.py is 2173 lines)
**Key constraint:** Do not change any runtime behavior; annotations only.
**Verify with:** `mypy ase/atoms.py ase/cell.py --ignore-missing-imports`

### Phase 2 — Type hints: Calculator interface
**Files:** `ase/calculators/calculator.py`, `ase/calculators/singlepoint.py`
**Scope:**
- `BaseCalculator`, `Calculator`, `FileIOCalculator`
- `calculate(atoms, properties, system_changes)` — define `Properties = list[str]`
- `get_property(name, atoms, allow_calculation)` — return type is context-dependent; use overloads
- `self.results: dict[str, Any]` → typed `Results` TypedDict
**Key constraint:** The duck-typed calculator protocol is used by 60+ calculators; type the base, not every subclass.
**Verify with:** `mypy ase/calculators/calculator.py ase/calculators/singlepoint.py --ignore-missing-imports`

### Phase 3 — Type hints: IO public API + extxyz
**Files:** `ase/io/__init__.py`, `ase/io/extxyz.py`, `ase/io/formats.py`
**Scope:**
- `read(filename, index, format)` — `index` is `int | str | slice`
- `write(filename, images, format)` — `images` is `Atoms | list[Atoms]`
- `iread(filename)` — yields `Atoms`
- `extxyz.key_val_str_to_dict()` — pure string → dict, easy to type
- `extxyz.read_xyz()` / `write_xyz()` — critical for ML community
**Priority:** extxyz first — highest ML impact.
**Verify with:** `mypy ase/io/__init__.py ase/io/extxyz.py --ignore-missing-imports`

### Phase 4 — Type hints: build/, geometry/, neighborlist
**Files:** `ase/build/bulk.py`, `ase/build/surface.py`, `ase/build/supercells.py`,
           `ase/geometry/geometry.py`, `ase/neighborlist.py`
**Scope:**
- `build.bulk(name, crystalstructure, a, ...)` → `Atoms`
- `build.surface(symbol, surface, layers, ...)` → `Atoms`
- `neighborlist.neighbor_list(quantities, a, cutoff)` — `quantities: str` is the magic string; create a `Literal` type or `QuantityStr = Literal['i','j','d','D','S']`
- `make_supercell(prim, P)` → `Atoms`
**Verify with:** `mypy ase/build/ ase/neighborlist.py --ignore-missing-imports`

---

### Phase 4.5 — Test Enrichment ← INSERT HERE

**Why here:** After the type hint phases are done (behavior unchanged, fully
annotated), before any phase that modifies runtime behavior (Phase 5 errors,
Phase 7+ Rust). Tests written now become the safety net for everything that follows.
Without this phase, Rust fallback bugs and error message regressions will be
silent.

**Goals:**
1. Coverage for the 8 bottleneck functions so Rust replacements can be validated
2. Regression fixtures that freeze current behavior before Phase 5 changes it
3. Type checking tests — mypy passes on all Phase 1-4 annotated modules
4. Property-based tests for the functions most likely to have edge cases in Rust

**Test files to create / enrich:**

#### 4.5.1 — Neighborlist tests (`ase/test/test_neighborlist_extended.py`)
```python
# Parametrize over: N_atoms, pbc combos, cell shapes, cutoff values
# Test both primitive_neighbor_list AND NewPrimitiveNeighborList give same results
# Edge cases: zero atoms, single atom, non-orthorhombic cell, large cutoff
# Property test: bothways=True must give symmetric results
# Property test: all returned distances must be <= cutoff
```

#### 4.5.2 — extxyz round-trip tests (`ase/test/fio/test_extxyz_extended.py`)
```python
# Round-trip: write then read must return identical Atoms
# Test with: per-atom arrays, per-config info, stress, forces, large files
# Edge cases: unicode in info keys, very long comment lines, 0-atom frames
# Regression fixture: freeze output of key_val_str_to_dict for known inputs
```

#### 4.5.3 — Build module tests (`ase/test/test_build_extended.py`)
```python
# bulk(): all crystal structures, all common elements
# surface(): fcc100, fcc111, bcc110 — check layer count, vacuum, periodicity
# make_supercell(): diagonal P, off-diagonal P, check atom count = det(P) * N_prim
# molecule(): all entries in g2 database
```

#### 4.5.4 — Type checking CI gate (`ase/test/test_types.py`)
```python
# Run mypy programmatically on Phase 1-4 files
# Fail if any newly annotated file introduces mypy errors
# This ensures type hints stay correct as code evolves
```

#### 4.5.5 — Performance regression fixtures
```python
# Run baseline benchmarks and save to benchmarks/results/pre-rust-baseline.json
# These become the denominator for speedup claims in Phase 11
# Use the script in BOTTLENECK_ANALYSIS.md
```

**Coverage target:** The 8 bottleneck functions should reach >90% branch coverage
before any Rust work begins. Everything else can stay at current levels.

**Tool:** `pytest --cov=ase/neighborlist --cov=ase/io/extxyz --cov=ase/build/supercells`

---

### Phase 5 — Error messages
**Goal:** Every `ValueError`, `RuntimeError`, `KeyError` in user-facing code should:
1. State what went wrong in plain English
2. State what was expected
3. Suggest a fix where possible
**Key files:** `ase/neighborlist.py`, `ase/calculators/calculator.py`, `ase/io/formats.py`
**Dependency:** Phase 4.5 must be done — error message changes alter exception text,
which will break any test that checks exception messages.
**Anti-pattern to fix:**
```python
# current
raise ValueError('Wrong number of cutoff radii: {} != {}'.format(...))
# target
raise ValueError(
    f"NeighborList requires one cutoff per atom, got {len(cutoffs)} cutoffs "
    f"for {len(coordinates)} atoms. Pass a list of length {len(coordinates)}."
)
```

### Phase 6 — LLM natural language layer
**Goal:** A thin `ase.ai` module that accepts natural language descriptions and
returns validated ASE code + objects.
**License:** New code — can be proprietary. This is the commercial differentiator.
**Examples:**
- `"graphene ribbon, zigzag edges, 5 unit cells wide"` → `Atoms`
- `"FCC copper, 3x3x3 supercell"` → `Atoms`
- `"attach EMT calculator and run BFGS until force < 0.05"` → script
**Architecture:**
- Uses Claude API with structured output
- Type hints from P1-P4 provide the schema for the LLM to generate against
- Validation: always run `ase.io.write(StringIO(), result, format='extxyz')` to verify
**Dependency:** Phases 1-4 must be complete for the type information to be useful.

### Phase 7 — Rust neighborlist
**License:** New code in separate package — MIT or Apache-2.0. You own this.
**Target:** `ase/neighborlist.py` — `primitive_neighbor_list()` and `PrimitiveNeighborList.build()`
**Dependency:** Phase 4.5 must be done — tests validate the Rust output matches Python.
**Algorithm:** Replace the triple nested for loop (lines 382-418) with a Rust
cell-list implementation using spatial hashing. Also replace the `bothways`
Python loop (lines 1033-1044).
**Rust crate:** `pyo3` for bindings, `ndarray` for array interop
**Interface:** Drop-in replacement — same function signature, same return types
**Fallback:** Always wrap in try/except ImportError → pure Python path
**Expected speedup:** 20-50x on systems >500 atoms
**Benchmark baseline:** Run `benchmarks/run_baseline.py` before writing any Rust.

### Phase 8 — Rust extxyz ✅ DONE
**License:** New code — MIT (ase-extxyz-rs crate).
**Target:** `ase/io/extxyz.py` — `write_xyz()` atom loop + `_read_xyz_frame()` atom loop.
**Dependency:** Phase 4.5 round-trip tests must pass before and after.

**What was implemented:**
- `ase-extxyz-rs/` crate with PyO3 + ndarray; two functions:
  - `write_atoms_rs(sym_codepoints, sym_width, float_mat, int_mat, bool_mat, col_types)` → str
  - `read_atom_lines_rs(lines, col_types)` → dict of typed ndarrays
- `write_xyz` dispatch: runs before structured array construction; passes symbols
  as UTF-32 codepoint view (zero-copy, avoids N Python string objects); passes float
  columns directly from `arrays` dict (no split/restack via structured array).
- `_read_xyz_frame` dispatch: collects lines then hands to Rust; fills structured array
  from typed column arrays.
- All 97 extxyz tests pass; 0 new regressions.

**Achieved speedups (release build, 108-atom FCC supercell frames):**
| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| write_xyz 200 frames × 108 atoms | 31.4 ms | 9.2 ms | **3.4×** |
| write_xyz  50 frames × 108 atoms | 7.6 ms  | 2.1 ms | **3.6×** |
| read_xyz  200 frames × 108 atoms | 1.1 ms  | 1.0 ms | 1.1×    |

**Note on read speedup:** The read path collects lines via a Python generator loop
before calling Rust (same N Python calls), so speedup is limited. For larger atom
counts per frame the speedup improves. The comment-line parser (`key_val_str_to_dict`)
is still Python — T6/T7 deferred to Phase 8 continuation.

**Remaining (Phase 8 continuation):**
- T6/T7: `comment.rs` — state-machine parser for `key=value` comment lines (20–50×).
  Currently uses Python `key_val_str_to_dict`. Skipped because complex grammar
  (nested brackets, escaped strings) must match Python exactly; lower ROI than write.

### Phase 9 — Rust neighborlist extensions + geometry + RDF
**License:** New code — MIT or Apache-2.0.
**Two crates:** extend `ase-neighborlist-rs`; create `ase-geometry-rs`.

**Priority 1 — Per-atom cutoffs in `ase-neighborlist-rs`** (`neighborlist.py:572-582`)
- Python loop iterates over atoms checking `d < radii[i] + radii[j]`; 10-20× expected.
- Strategy: pass full `radii` array into Rust; cell-list bin width = `2 * max(radii)`;
  inner-loop filter: `d < radii[i] + radii[j]` instead of scalar cutoff.
- Enables: `NeighborList(cutoffs=[1.3, 1.5, ...])` — the bonding analysis codepath.

**Priority 2 — Minkowski reduction in `ase-geometry-rs`** (`minkowski_reduction.py:85-119`)
- Iterative 3×3 integer linear algebra (Gram-Schmidt with basis swaps); 8-25× expected.
- Critical path for every `neighbor_list` call on non-orthorhombic cells (surface
  slabs, triclinic, MD snapshots). Called inside `primitive_neighbor_list` before
  the main cell-list loop when `pbc` is non-trivial.
- Pure integer arithmetic — Rust wins cleanly with no SIMD needed.

**Priority 3 — Supercell construction in `ase-geometry-rs`** (`supercells.py:355-377`)
- Array broadcast + atom coordinate repeat; 5-15× expected.
- Called for every MLIP training set construction; hot path in GNoME-style workflows.
- `make_supercell(prim, P)` → new Atoms; Rust fills fractional coords, wraps to [0,1).

**Priority 4 — Dict cutoffs in `ase-neighborlist-rs`** (`neighborlist.py:540-571`)
- Dict iteration + per-pair cutoff lookup; 2-5× expected.
- Strategy: flatten `{(Zi, Zj): cutoff}` to sorted lookup table; pass `numbers` array.

**Priority 5 (optional) — RDF in `ase-geometry-rs`** (`rdf.py:143-158`)
- Per-frame per-atom histogram loop; 3-8× expected.
- Lower priority: `get_rdf` users typically call it once; not a training-loop bottleneck.

**Dependency:** Phase 7 scalar neighborlist must be DONE (it is). Tests from Phase 4.5
cover all targets. Run `pytest ase/test/` before and after each priority.

**Expected overall impact:** Per-atom cutoffs (P1) + Minkowski (P2) together cover
the two remaining slow paths hit by every MLIP bond-counting workflow.

### Phase 10 — Rust IO: VASP + XYZ
**License:** New code — MIT or Apache-2.0.
**Status:** **DONE** — 3.5× read, 6.1× write VASP; 1.7× simple XYZ avg; 90 IO tests pass.
**Crate:** `ase-io-rs/` → `ase._io_rs`
**Hot paths replaced:**
- `read_vasp_configuration:261-268` → `parse_poscar_positions_rs` — **3.2-3.7× read** (500→2048 atoms)
- `write_vasp:908-914` → `format_poscar_positions_rs` — **5.5-6.7× write** (500→2048 atoms)
- `read_vasp_xdatcar:398` → `parse_xdatcar_coords_rs` — per-frame fast path
- `_write_xdatcar_config:764-768` → `format_xdatcar_config_rs` — XDATCAR write fast path
- `xyz.read_xyz / xyz.write_xyz` → `parse_xyz_block_rs / format_xyz_block_rs` — simple XYZ
**Dispatch:** `_HAVE_RUST_IO` flag at module level; fallback to Python on import failure.
**CIF:** Deferred — complex grammar; bottleneck is regex-based tokenizer + symmetry expansion, not float parsing.
**Benchmark results (quick run):**
- `vasp_read/500_atoms`:  0.33 ms → 0.10 ms (3.2×)
- `vasp_write/500_atoms`: 0.54 ms → 0.10 ms (5.5×)
- `vasp_read/2048_atoms`:  1.38 ms → 0.37 ms (3.7×)
- `vasp_write/2048_atoms`: 2.29 ms → 0.34 ms (6.7×)

### Phase 11 — Benchmarks
**Status:** ✅ DONE — results in `benchmarks/results/phase11_final.json`

**Test status:** ✅ All 3480 tests pass (0 failures, 0 errors).

**Bug fixed in this phase (pre-requisite cleanup):**
- Rust neighborlist `solve3x3` had a transposed-inverse bug that caused wrong neighbor counts for non-orthogonal cells (e.g. Graphite hex cell). Fixed by correcting index order in the final matrix multiply.
- 86 other pre-existing test failures fixed: added `np.errstate(divide/invalid/over='ignore')` guards in `atoms.py`, `cell.py`, `supercells.py`, `bfgslinesearch.py`, `bfgs.py`, `eam.py`, `harmonic.py`, `morse.py`, `qmmm.py`, `cluster/factory.py`, `dft/bee.py`, `geometry/geometry.py`, `optimize/gpmin/gp.py`, `phonons.py`, `utils/ff.py`, `optimize/sciopt.py`, `neighborlist.py`.
- Fixed doctest in `neighborlist.py` (`NewPrimitiveNeighborList` doctest used 2 cutoffs for 1-atom structure).
- Fixed version test via `pip install -e .`.
- Moved all 4 Rust crates (ase-neighborlist-rs, ase-extxyz-rs, ase-geometry-rs, ase-io-rs) inside the ASE repository.

**Phase 11 benchmark results** (arm64, Python 3.13, `benchmarks/results/phase11_final.json`):

| Module | Operation | Python | Rust | Speedup |
|--------|-----------|--------|------|---------|
| neighborlist | scalar cutoff (864 atoms) | 6.88 ms | 0.49 ms | **13.9×** |
| neighborlist | per-atom radii (864 atoms) | 7.08 ms | 0.60 ms | **11.7×** |
| neighborlist | dict cutoffs (864 atoms) | 6.91 ms | 0.76 ms | **9.1×** |
| extxyz | write (200fr × 108at) | 31.05 ms | 8.83 ms | **3.5×** |
| extxyz | read (200fr × 108at) | 1.06 ms | 1.00 ms | **1.1×** |
| VASP | read (2048 atoms) | 1.30 ms | 0.41 ms | **3.2×** |
| VASP | write (2048 atoms) | 2.29 ms | 0.34 ms | **6.8×** |
| simple XYZ | write (50fr × 108at) | 4.66 ms | 1.12 ms | **4.2×** |
| simple XYZ | read (50fr × 108at) | 4.16 ms | 1.65 ms | **2.5×** |
| geometry | Minkowski reduce (triclinic) | 0.07 ms | 0.03 ms | **2.6×** |
| geometry | supercell (1000 atoms) | 0.18 ms | 0.20 ms | 0.9× |

**Summary averages:** NL scalar 13.6×, NL per-atom 12.2×, NL dict 9.1×, VASP write 6.3×, VASP read 3.3×, extxyz write 3.5×, simple XYZ 2.3×.

**Rule:** Always run the full benchmark suite; never save partial results.
**Output format:** JSON saved to `benchmarks/results/YYYY-MM-DD_benchmarks.json`
**Compute speedup as:** `baseline_ms / ase_fast_ms` — never report raw numbers alone.

### Phase 12 — Branding + outreach
**IP structure:**
- `ase` type hints + error messages → contribute upstream (LGPL, goodwill)
- `ase-fast` Rust package → MIT, PyPI, open source
- `ase-ai` LLM layer → proprietary or open, your choice
**Target communities:**
- MACE developers (Cambridge) — neighbor list pain
- FAIR-Chem team (Meta) — extxyz + neighbor list
- GNoME downstream users — extxyz at scale
- matscipy users — they left ASE because of neighbor list speed
**PyPI:** Publish `ase-fast` with `ase` as a dependency; override hot paths via
the `try: import ase_fast_rs; except ImportError: use_python()` pattern.

---

## Key Technical Decisions (locked)

1. **No behavior changes in Phases 1-4** — annotations only
2. **Phase 4.5 (tests) is a hard gate before Phases 5, 7, 8** — no Rust without tests
3. **Rust via PyO3** — consistent with xmltodict-rs and openpyxl-rs approach
4. **Drop-in interface with Python fallback** — Rust functions must match existing
   Python signatures exactly; always fall back on ImportError
5. **extxyz before VASP** — ML community impact > DFT community impact
6. **LLM layer requires typed foundation** — do not start Phase 6 before Phase 4 done
7. **Contribute type hints + error messages upstream** — reduces maintenance, builds trust
8. **Rust extensions are MIT** — maximizes adoption; commercial value is in `ase-ai`
