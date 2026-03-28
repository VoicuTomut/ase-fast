#!/usr/bin/env python
"""
ASE-fast benchmark suite — baseline (pure Python) vs ase-fast (Rust).

Each section runs the same operation twice:
  - with Rust disabled (_HAVE_RUST_* = False)  → baseline
  - with Rust enabled  (_HAVE_RUST_* = True)   → ase-fast

Reports speedup = baseline_ms / ase_fast_ms.

Results are written to  benchmarks/results/YYYY-MM-DD_HHhMM.json

Usage:
    python benchmarks/run_benchmarks.py            # full suite
    python benchmarks/run_benchmarks.py --output path/to/out.json
"""
from __future__ import annotations

import argparse
import io as _io
import json
import sys
import time
import tracemalloc
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from ase.build import bulk, fcc111, molecule, make_supercell
from ase.geometry import get_distances, wrap_positions, minkowski_reduce
from ase.io.extxyz import read_xyz, write_xyz
from ase.io.vasp import read_vasp, write_vasp
import ase.io.xyz as _xyz_simple
from ase import neighborlist as _nl
from ase.neighborlist import primitive_neighbor_list, neighbor_list
import ase.io.extxyz as _extxyz
import ase.geometry.minkowski_reduction as _mink
import ase.build.supercells as _sc
import ase.io.vasp as _vasp


# ─── Timing / memory helpers ────────────────────────────────────────────────

def _median_s(fn: Callable, n: int) -> float:
    """Median wall-clock time in seconds over n calls (warnings suppressed)."""
    times: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[n // 2]


def _peak_kb(fn: Callable) -> float:
    """Peak memory increment in KB for a single call (warnings suppressed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return peak / 1024


def _bench(
    label: str,
    fn: Callable,
    n: int,
    *,
    rust_fn: Callable | None = None,
) -> dict[str, Any]:
    """
    Time and profile *fn*.  If *rust_fn* is provided (same operation via
    the Rust path), also time that and compute the speedup.
    """
    t_py = _median_s(fn, n) * 1e3       # ms
    m_py = _peak_kb(fn)

    result: dict[str, Any] = {
        "label": label,
        "python_median_ms": round(t_py, 3),
        "python_peak_kb":   round(m_py, 1),
    }

    rust_available = (
        _nl._HAVE_RUST_NEIGHBORLIST
        if rust_fn is not None
        else _extxyz._HAVE_RUST_EXTXYZ
    )
    if rust_fn is not None and rust_available:
        # Warm up then measure
        rust_fn()
        t_rs = _median_s(rust_fn, n) * 1e3
        m_rs = _peak_kb(rust_fn)
        speedup = t_py / t_rs if t_rs > 0 else float("inf")
        result.update({
            "rust_median_ms": round(t_rs, 3),
            "rust_peak_kb":   round(m_rs, 1),
            "speedup_x":      round(speedup, 1),
        })

    return result


def _fmt(result: dict[str, Any]) -> str:
    label = result["label"]
    t_py  = result["python_median_ms"]
    m_py  = result["python_peak_kb"]
    if "rust_median_ms" in result:
        t_rs = result["rust_median_ms"]
        m_rs = result["rust_peak_kb"]
        sp   = result["speedup_x"]
        return (
            f"  {label:<45s}  {t_py:>8.2f} ms  {m_py:>9.1f} KB"
            f"  │ Rust {t_rs:>7.2f} ms  {m_rs:>8.1f} KB  {sp:>6.1f}×"
        )
    return (
        f"  {label:<45s}  {t_py:>8.2f} ms  {m_py:>9.1f} KB"
    )


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _fcc(n: int):
    """FCC Cu supercell with approximately n atoms."""
    repeat = max(1, round((n / 4) ** (1 / 3)))
    return bulk('Cu', 'fcc', a=3.615, cubic=True).repeat(repeat)


def _h2o_frames(n: int):
    return [molecule('H2O')] * n


def _fcc_frames(n_frames: int, n_atoms: int = 108):
    """FCC Cu supercell frames (~n_atoms atoms each) for ML-scale benchmarks."""
    atoms = _fcc(n_atoms)
    return [atoms] * n_frames


def _xyz_buf(n_frames: int, atoms_per_frame: int = 3) -> _io.StringIO:
    """Pre-write n_frames to a StringIO buffer for read benchmarks."""
    buf = _io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if atoms_per_frame <= 3:
            write_xyz(buf, _h2o_frames(n_frames))
        else:
            write_xyz(buf, _fcc_frames(n_frames, atoms_per_frame))
    buf.seek(0)
    return buf


def _poscar_buf(n_atoms: int) -> _io.StringIO:
    """Write a POSCAR to StringIO and return the buffer (for read benchmarks)."""
    atoms = _fcc(n_atoms)
    buf = _io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_vasp(buf, atoms)
    buf.seek(0)
    return buf


def _simple_xyz_buf(n_frames: int, atoms_per_frame: int = 3) -> _io.StringIO:
    """Pre-write n_frames using the *simple* xyz module (not extxyz)."""
    buf = _io.StringIO()
    frames = _h2o_frames(n_frames) if atoms_per_frame <= 3 else _fcc_frames(n_frames, atoms_per_frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _xyz_simple.write_xyz(buf, frames)
    buf.seek(0)
    return buf


# ─── Benchmark definitions ───────────────────────────────────────────────────

def _make_neighborlist_benches(atoms, label_suffix: str, n: int):
    """Return (label, py_fn, rust_fn, n) for a given atoms object."""
    def py_fn():
        _nl._HAVE_RUST_NEIGHBORLIST = False
        primitive_neighbor_list('ijdDS', atoms.pbc, atoms.cell, atoms.positions, 3.0)
        _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG

    def rs_fn():
        _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG
        primitive_neighbor_list('ijdDS', atoms.pbc, atoms.cell, atoms.positions, 3.0)

    return (f"neighborlist/{label_suffix}", py_fn, n, rs_fn)


_RUST_ORIG = _nl._HAVE_RUST_NEIGHBORLIST


def _run_all(quick: bool) -> list[dict[str, Any]]:
    reps = 3 if quick else 7
    reps_heavy = 1 if quick else 3

    results = []

    # ── Section 1: Neighborlist (scalar cutoff) ─────────────────────────────────
    print("\n  ── Neighborlist (scalar cutoff) ─────────────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    for n_atoms, n_reps in [(500, reps), (1000, reps), (4000, reps_heavy)]:
        atoms = _fcc(n_atoms)
        label, py_fn, nr, rs_fn = _make_neighborlist_benches(
            atoms, f"{len(atoms)}_atoms", n_reps
        )
        r = _bench(label, py_fn, nr, rust_fn=rs_fn)
        results.append(r)
        print(_fmt(r))

    # ── Section 2: extxyz IO (write + read) ──────────────────
    print("\n  ── extxyz IO (write + read) ────────────────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _EXTXYZ_ORIG = _extxyz._HAVE_RUST_EXTXYZ

    def _make_extxyz_benches(label, frames, n_frames, atoms_per_frame):
        buf = _io.StringIO()

        def write_py(f=frames, b=buf):
            _extxyz._HAVE_RUST_EXTXYZ = False
            b.seek(0); b.truncate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_xyz(b, f)
            _extxyz._HAVE_RUST_EXTXYZ = _EXTXYZ_ORIG

        def write_rs(f=frames, b=buf):
            _extxyz._HAVE_RUST_EXTXYZ = _EXTXYZ_ORIG
            b.seek(0); b.truncate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_xyz(b, f)

        pre_written = _xyz_buf(n_frames, atoms_per_frame)

        def read_py(b=pre_written):
            _extxyz._HAVE_RUST_EXTXYZ = False
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            list(read_xyz(b))
            _extxyz._HAVE_RUST_EXTXYZ = _EXTXYZ_ORIG

        def read_rs(b=pre_written):
            _extxyz._HAVE_RUST_EXTXYZ = _EXTXYZ_ORIG
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            list(read_xyz(b))

        return (
            (f"write_xyz/{label}", write_py, write_rs),
            (f"read_xyz/{label}",  read_py,  read_rs),
        )

    # Small molecules (H2O, 3 atoms) — tests overhead regime
    for n_frames in ([50, 200] if quick else [50, 200, 1000]):
        label = f"{n_frames}fr_3at"
        frames = _h2o_frames(n_frames)
        for (lbl, py_fn, rs_fn) in _make_extxyz_benches(label, frames, n_frames, 3):
            r = _bench(lbl, py_fn, reps, rust_fn=rs_fn)
            results.append(r)
            print(_fmt(r))

    # ML-scale (108-atom FCC supercell) — real MLIP training workload
    for n_frames in ([10, 50] if quick else [10, 50, 200]):
        label = f"{n_frames}fr_108at"
        frames = _fcc_frames(n_frames, 108)
        for (lbl, py_fn, rs_fn) in _make_extxyz_benches(label, frames, n_frames, 108):
            r = _bench(lbl, py_fn, reps, rust_fn=rs_fn)
            results.append(r)
            print(_fmt(r))

    # ── Section 3: Geometry utilities (no Rust yet) ──────────────────────
    print("\n  ── Geometry utilities ─────────────────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Median':>9s}  {'Peak RAM':>9s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}")

    atoms_500 = _fcc(500)

    def wrap_fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        wrap_positions(atoms_500.positions, atoms_500.cell, atoms_500.pbc)

    r = _bench("wrap_positions/500_atoms", wrap_fn, reps)
    results.append(r); print(_fmt(r))

    def dist_fn():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        p = atoms_500.positions
        get_distances(p[:250], p[250:], cell=atoms_500.cell, pbc=atoms_500.pbc)

    r = _bench("get_distances/500_atoms", dist_fn, reps)
    results.append(r); print(_fmt(r))

    # ── Section 4: per-atom & dict cutoffs ─────────────────────
    print("\n  ── Neighborlist (per-atom & dict cutoffs) ─────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _RUST_ORIG = _nl._HAVE_RUST_NEIGHBORLIST
    for n_atoms, n_reps in [(500, reps), (1000, reps)]:
        atoms_b = _fcc(n_atoms)
        _radii = [1.5] * len(atoms_b)
        def _py_radii(a=atoms_b, r=_radii):
            _nl._HAVE_RUST_NEIGHBORLIST = False
            neighbor_list('ij', a, r)
            _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG
        def _rs_radii(a=atoms_b, r=_radii):
            _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG
            neighbor_list('ij', a, r)
        rb = _bench(f"neighborlist_radii/{len(atoms_b)}_atoms", _py_radii, n_reps, rust_fn=_rs_radii)
        results.append(rb); print(_fmt(rb))

        _dcut = {('Cu', 'Cu'): 3.0}
        def _py_dict(a=atoms_b, c=_dcut):
            _nl._HAVE_RUST_NEIGHBORLIST = False
            neighbor_list('ij', a, c)
            _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG
        def _rs_dict(a=atoms_b, c=_dcut):
            _nl._HAVE_RUST_NEIGHBORLIST = _RUST_ORIG
            neighbor_list('ij', a, c)
        rd = _bench(f"neighborlist_dict/{len(atoms_b)}_atoms", _py_dict, n_reps, rust_fn=_rs_dict)
        results.append(rd); print(_fmt(rd))

    # ── Section 5: Minkowski reduction ─────────────────────────
    print("\n  ── Minkowski reduction ──────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _RUST_GEOM_ORIG = _mink._HAVE_RUST_GEOM
    # Triclinic cell — worst case for Minkowski (many iterations needed)
    import numpy as np
    _triclinic_cell = np.array([[5.0, 2.3, 1.1], [0.0, 4.8, 1.9], [0.0, 0.0, 3.7]])

    def _py_mink(c=_triclinic_cell):
        _mink._HAVE_RUST_GEOM = False
        minkowski_reduce(c)
        _mink._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
    def _rs_mink(c=_triclinic_cell):
        _mink._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
        minkowski_reduce(c)
    rm = _bench("minkowski_reduce/triclinic", _py_mink, reps, rust_fn=_rs_mink)
    results.append(rm); print(_fmt(rm))

    # FCC primitive cell (already reduced — fast path)
    _fcc_cell = np.array([[0.0, 1.8075, 1.8075], [1.8075, 0.0, 1.8075], [1.8075, 1.8075, 0.0]])
    def _py_mink_fcc(c=_fcc_cell):
        _mink._HAVE_RUST_GEOM = False
        minkowski_reduce(c)
        _mink._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
    def _rs_mink_fcc(c=_fcc_cell):
        _mink._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
        minkowski_reduce(c)
    rmc = _bench("minkowski_reduce/fcc_primitive", _py_mink_fcc, reps, rust_fn=_rs_mink_fcc)
    results.append(rmc); print(_fmt(rmc))

    # ── Section 6: Supercell construction ───────────────────────
    print("\n  ── Supercell construction ───────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _prim_cu = bulk('Cu', 'fcc', a=3.615)
    for sc_size, n_reps in [(5, reps), (10, reps_heavy)]:
        _P = np.diag([sc_size, sc_size, sc_size])
        def _py_sc(p=_prim_cu, P=_P):
            _sc._HAVE_RUST_GEOM = False
            make_supercell(p, P)
            _sc._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
        def _rs_sc(p=_prim_cu, P=_P):
            _sc._HAVE_RUST_GEOM = _RUST_GEOM_ORIG
            make_supercell(p, P)
        n_sc = sc_size**3 * len(_prim_cu)
        rs = _bench(f"make_supercell/{n_sc}_atoms", _py_sc, n_reps, rust_fn=_rs_sc)
        results.append(rs); print(_fmt(rs))

    # ── Section 7: VASP IO ─────────────────────────────────────
    print("\n  ── VASP IO ─────────────────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _RUST_IO_ORIG = getattr(_vasp, '_HAVE_RUST_IO', False)

    for n_atoms, n_reps in ([(500, reps), (2000, reps_heavy)] if quick else [(500, reps), (2000, reps), (5000, reps_heavy)]):
        atoms_v = _fcc(n_atoms)
        _poscar = _poscar_buf(n_atoms)

        def _py_vread(b=_poscar):
            _vasp._HAVE_RUST_IO = False
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            read_vasp(b)
            _vasp._HAVE_RUST_IO = _RUST_IO_ORIG

        def _rs_vread(b=_poscar):
            _vasp._HAVE_RUST_IO = _RUST_IO_ORIG
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            read_vasp(b)

        def _py_vwrite(a=atoms_v):
            _vasp._HAVE_RUST_IO = False
            buf = _io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_vasp(buf, a)
            _vasp._HAVE_RUST_IO = _RUST_IO_ORIG

        def _rs_vwrite(a=atoms_v):
            _vasp._HAVE_RUST_IO = _RUST_IO_ORIG
            buf = _io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                write_vasp(buf, a)

        n_actual = len(atoms_v)
        # read benchmark — rust_fn=None until Rust is implemented (Python baseline only)
        rust_read_fn = _rs_vread if _RUST_IO_ORIG else None
        rust_write_fn = _rs_vwrite if _RUST_IO_ORIG else None
        rr = _bench(f"vasp_read/{n_actual}_atoms", _py_vread, n_reps, rust_fn=rust_read_fn)
        results.append(rr); print(_fmt(rr))
        rw = _bench(f"vasp_write/{n_actual}_atoms", _py_vwrite, n_reps, rust_fn=rust_write_fn)
        results.append(rw); print(_fmt(rw))

    # ── Section 8: Simple XYZ IO ───────────────────────────────
    print("\n  ── Simple XYZ IO ───────────────────────────────────────────")
    print(f"  {'Label':<45s}  {'Python':>9s}  {'Py RAM':>9s}  │ {'Rust':>9s}  {'Rs RAM':>8s}  {'Speed':>7s}")
    print(f"  {'-'*45}  {'-'*9}  {'-'*9}  │ {'-'*9}  {'-'*8}  {'-'*7}")

    _RUST_IO_ORIG = getattr(_vasp, '_HAVE_RUST_IO', False)
    _RUST_XYZ_ORIG = getattr(_xyz_simple, '_HAVE_RUST_IO', False)

    for (n_frames, n_at, label_suffix) in (
        [(50, 3, "50fr_3at"), (200, 3, "200fr_3at")] if quick
        else [(50, 3, "50fr_3at"), (200, 3, "200fr_3at"), (1000, 3, "1000fr_3at"),
              (10, 108, "10fr_108at"), (50, 108, "50fr_108at")]
    ):
        frames = _h2o_frames(n_frames) if n_at <= 3 else _fcc_frames(n_frames, n_at)
        pre_written = _simple_xyz_buf(n_frames, n_at)

        def _py_swrite(f=frames):
            _xyz_simple._HAVE_RUST_IO = False
            buf = _io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _xyz_simple.write_xyz(buf, f)
            _xyz_simple._HAVE_RUST_IO = _RUST_XYZ_ORIG

        def _rs_swrite(f=frames):
            _xyz_simple._HAVE_RUST_IO = _RUST_XYZ_ORIG
            buf = _io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _xyz_simple.write_xyz(buf, f)

        def _py_sread(b=pre_written):
            _xyz_simple._HAVE_RUST_IO = False
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            list(_xyz_simple.read_xyz(b, slice(None)))
            _xyz_simple._HAVE_RUST_IO = _RUST_XYZ_ORIG

        def _rs_sread(b=pre_written):
            _xyz_simple._HAVE_RUST_IO = _RUST_XYZ_ORIG
            b.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            list(_xyz_simple.read_xyz(b, slice(None)))

        rust_w = _rs_swrite if _RUST_XYZ_ORIG else None
        rust_r = _rs_sread if _RUST_XYZ_ORIG else None
        rsw = _bench(f"xyz_simple_write/{label_suffix}", _py_swrite, reps, rust_fn=rust_w)
        results.append(rsw); print(_fmt(rsw))
        rsr = _bench(f"xyz_simple_read/{label_suffix}", _py_sread, reps, rust_fn=rust_r)
        results.append(rsr); print(_fmt(rsr))

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ASE OldGold unified benchmarks")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer repeats for a fast smoke-test run")
    ts = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    default_out = str(_REPO_ROOT / "benchmarks" / "results" / f"{ts}_benchmarks.json")
    parser.add_argument("--output", default=default_out,
                        help="Output JSON path (default: benchmarks/results/<timestamp>.json)")
    args = parser.parse_args()

    nl_status = "ACTIVE ✓" if _nl._HAVE_RUST_NEIGHBORLIST else "not found (Python fallback)"
    xyz_status = "ACTIVE ✓" if _extxyz._HAVE_RUST_EXTXYZ else "not found (Python fallback)"
    print(f"\n{'='*80}")
    print("  ASE OldGold — Unified Benchmark Suite")
    print(f"  Rust neighborlist extension: {nl_status}")
    print(f"  Rust extxyz extension:       {xyz_status}")
    print(f"  Mode: {'quick' if args.quick else 'full'}")
    print(f"{'='*80}")

    results = _run_all(quick=args.quick)

    print(f"\n{'='*80}")

    # Summary: speedup lines only
    nl_scalar_rows = [r for r in results if "speedup_x" in r and r["label"].startswith("neighborlist/")]
    nl_radii_rows  = [r for r in results if "speedup_x" in r and r["label"].startswith("neighborlist_radii")]
    nl_dict_rows   = [r for r in results if "speedup_x" in r and r["label"].startswith("neighborlist_dict")]
    xyz_rows       = [r for r in results if "speedup_x" in r and r["label"].startswith(("write_xyz", "read_xyz"))]
    mink_rows      = [r for r in results if "speedup_x" in r and r["label"].startswith("minkowski")]
    sc_rows        = [r for r in results if "speedup_x" in r and r["label"].startswith("make_supercell")]
    if nl_scalar_rows:
        avg = sum(r["speedup_x"] for r in nl_scalar_rows) / len(nl_scalar_rows)
        print(f"\n  Rust neighborlist scalar cutoff avg speedup: {avg:.1f}×")
    if nl_radii_rows:
        avg = sum(r["speedup_x"] for r in nl_radii_rows) / len(nl_radii_rows)
        print(f"  Rust neighborlist per-atom radii avg speedup: {avg:.1f}×")
    if nl_dict_rows:
        avg = sum(r["speedup_x"] for r in nl_dict_rows) / len(nl_dict_rows)
        print(f"  Rust neighborlist dict cutoff avg speedup: {avg:.1f}×")
    if xyz_rows:
        avg = sum(r["speedup_x"] for r in xyz_rows) / len(xyz_rows)
        print(f"\n  Rust extxyz average speedup: {avg:.1f}×")
        print(f"  (write + read atom loops; comment-line parser still Python)")
    if mink_rows:
        avg = sum(r["speedup_x"] for r in mink_rows) / len(mink_rows)
        print(f"\n  Rust Minkowski reduction avg speedup: {avg:.1f}×")
    if sc_rows:
        avg = sum(r["speedup_x"] for r in sc_rows) / len(sc_rows)
        print(f"  Rust supercell construction avg speedup: {avg:.1f}×")

    vasp_read_rows  = [r for r in results if "speedup_x" in r and r["label"].startswith("vasp_read")]
    vasp_write_rows = [r for r in results if "speedup_x" in r and r["label"].startswith("vasp_write")]
    xyz_simple_rows = [r for r in results if "speedup_x" in r and r["label"].startswith("xyz_simple")]
    if vasp_read_rows:
        avg = sum(r["speedup_x"] for r in vasp_read_rows) / len(vasp_read_rows)
        print(f"\n  Rust VASP read avg speedup: {avg:.1f}×")
    if vasp_write_rows:
        avg = sum(r["speedup_x"] for r in vasp_write_rows) / len(vasp_write_rows)
        print(f"  Rust VASP write avg speedup: {avg:.1f}×")
    if xyz_simple_rows:
        avg = sum(r["speedup_x"] for r in xyz_simple_rows) / len(xyz_simple_rows)
        print(f"  Rust simple XYZ avg speedup: {avg:.1f}×")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "rust_available": _nl._HAVE_RUST_NEIGHBORLIST,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results written to {out}\n")


if __name__ == "__main__":
    main()
