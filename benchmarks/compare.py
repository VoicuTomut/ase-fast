#!/usr/bin/env python
"""
ASE-fast — speed comparison: baseline (Python) vs ase-fast (Rust).

Produces:
  - A rich terminal table with timing for every operation
  - benchmarks/results/compare_<timestamp>.json  (machine-readable)
  - benchmarks/results/compare_<timestamp>.md    (paste-ready GitHub table)

Usage:
    python benchmarks/compare.py
    python benchmarks/compare.py --output results/my_machine.json
"""
from __future__ import annotations

import argparse
import io as _io
import json
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import ase.neighborlist as _nl
import ase.io.extxyz as _extxyz
import ase.geometry.minkowski_reduction as _mink
import ase.build.supercells as _sc
import ase.io.vasp as _vasp
import ase.io.xyz as _xyz

from ase.build import bulk, make_supercell
from ase.geometry import minkowski_reduce
from ase.neighborlist import primitive_neighbor_list, neighbor_list
from ase.io.extxyz import read_xyz, write_xyz
from ase.io.vasp import read_vasp, write_vasp

# ── helpers ──────────────────────────────────────────────────────────────────

def _time_ms(fn, n=7):
    """Median wall-clock time in ms over n calls (warm-up discarded)."""
    fn()  # warm-up
    times = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[n // 2]


@contextmanager
def _rust_off(module, flag):
    orig = getattr(module, flag)
    setattr(module, flag, False)
    try:
        yield
    finally:
        setattr(module, flag, orig)


def _measure(label, py_fn, rs_fn, n=7):
    """Run both paths and return a result dict."""
    with _rust_off(_nl, '_HAVE_RUST_NEIGHBORLIST'), \
         _rust_off(_extxyz, '_HAVE_RUST_EXTXYZ'), \
         _rust_off(_mink, '_HAVE_RUST_GEOM'), \
         _rust_off(_sc, '_HAVE_RUST_GEOM'), \
         _rust_off(_vasp, '_HAVE_RUST_IO'), \
         _rust_off(_xyz, '_HAVE_RUST_IO'):
        py_ms = _time_ms(py_fn, n)

    rs_ms = _time_ms(rs_fn, n)

    speedup = round(py_ms / rs_ms, 1) if rs_ms > 0 else 0

    return {
        "label":     label,
        "py_ms":     round(py_ms, 3),
        "rs_ms":     round(rs_ms, 3),
        "speedup_x": speedup,
    }


# ── atom builders ────────────────────────────────────────────────────────────

def _fcc(n):
    repeat = max(1, round((n / 4) ** (1 / 3)))
    return bulk('Cu', 'fcc', a=3.615, cubic=True).repeat(repeat)


def _poscar_buf(n):
    atoms = _fcc(n)
    buf = _io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_vasp(buf, atoms)
    buf.seek(0)
    return buf


def _xyz_buf(n_frames, n_atoms):
    atoms = _fcc(n_atoms)
    buf = _io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_xyz(buf, [atoms] * n_frames)
    buf.seek(0)
    return buf


# ── benchmark definitions ────────────────────────────────────────────────────

def run_all(n_reps=7):
    results = []

    # ── 1. Neighbor list — scalar cutoff ─────────────────────────────────────
    print("\n  Neighbor list — scalar cutoff")
    for n_atoms in [500, 864, 4000]:
        atoms = _fcc(n_atoms)
        nr = n_reps if n_atoms < 2000 else max(3, n_reps // 2)

        def py(a=atoms): primitive_neighbor_list('ijdDS', a.pbc, a.cell, a.positions, 3.0)
        def rs(a=atoms): primitive_neighbor_list('ijdDS', a.pbc, a.cell, a.positions, 3.0)

        r = _measure(f"nl_scalar/{n_atoms}_atoms", py, rs, nr)
        results.append(r); _print_row(r)

    # ── 2. Neighbor list — per-atom radii ─────────────────────────────────────
    print("\n  Neighbor list — per-atom radii")
    for n_atoms in [500, 864]:
        atoms = _fcc(n_atoms)
        radii = [1.5] * len(atoms)

        def py(a=atoms, r=radii): neighbor_list('ij', a, r)
        def rs(a=atoms, r=radii): neighbor_list('ij', a, r)

        r = _measure(f"nl_radii/{n_atoms}_atoms", py, rs, n_reps)
        results.append(r); _print_row(r)

    # ── 3. Neighbor list — dict cutoffs ──────────────────────────────────────
    print("\n  Neighbor list — dict cutoffs")
    for n_atoms in [500, 864]:
        atoms = _fcc(n_atoms)
        cuts  = {('Cu', 'Cu'): 3.0}

        def py(a=atoms, c=cuts): neighbor_list('ij', a, c)
        def rs(a=atoms, c=cuts): neighbor_list('ij', a, c)

        r = _measure(f"nl_dict/{n_atoms}_atoms", py, rs, n_reps)
        results.append(r); _print_row(r)

    # ── 4. extxyz write ───────────────────────────────────────────────────────
    print("\n  extxyz write")
    for n_frames, n_at in [(50, 108), (200, 108), (1000, 3)]:
        atoms = _fcc(n_at)

        def py(a=atoms, nf=n_frames):
            buf = _io.StringIO()
            write_xyz(buf, [a] * nf)
        def rs(a=atoms, nf=n_frames):
            buf = _io.StringIO()
            write_xyz(buf, [a] * nf)

        lbl = f"extxyz_write/{n_frames}fr_{n_at}at"
        r = _measure(lbl, py, rs, n_reps)
        results.append(r); _print_row(r)

    # ── 5. extxyz read ────────────────────────────────────────────────────────
    print("\n  extxyz read")
    for n_frames, n_at in [(50, 108), (200, 108)]:
        buf_data = _xyz_buf(n_frames, n_at).getvalue()

        def py(d=buf_data): list(read_xyz(_io.StringIO(d)))
        def rs(d=buf_data): list(read_xyz(_io.StringIO(d)))

        lbl = f"extxyz_read/{n_frames}fr_{n_at}at"
        r = _measure(lbl, py, rs, n_reps)
        results.append(r); _print_row(r)

    # ── 6. VASP write ─────────────────────────────────────────────────────────
    print("\n  VASP write")
    for n_atoms in [500, 2048, 5324]:
        atoms = _fcc(n_atoms)
        nr = n_reps if n_atoms < 3000 else max(3, n_reps // 2)

        def py(a=atoms):
            buf = _io.StringIO()
            write_vasp(buf, a)
        def rs(a=atoms):
            buf = _io.StringIO()
            write_vasp(buf, a)

        r = _measure(f"vasp_write/{n_atoms}_atoms", py, rs, nr)
        results.append(r); _print_row(r)

    # ── 7. VASP read ──────────────────────────────────────────────────────────
    print("\n  VASP read")
    for n_atoms in [500, 2048, 5324]:
        poscar = _poscar_buf(n_atoms).getvalue()
        nr = n_reps if n_atoms < 3000 else max(3, n_reps // 2)

        def py(d=poscar): read_vasp(_io.StringIO(d))
        def rs(d=poscar): read_vasp(_io.StringIO(d))

        r = _measure(f"vasp_read/{n_atoms}_atoms", py, rs, nr)
        results.append(r); _print_row(r)

    # ── 8. Simple XYZ write / read ───────────────────────────────────────────
    print("\n  Simple XYZ write / read")
    from ase.io.xyz import write_xyz as _write_simple_xyz, read_xyz as _read_simple_xyz
    for n_frames, n_at in [(50, 108), (200, 108)]:
        atoms = _fcc(n_at)
        frames = [atoms] * n_frames

        def py_w(fs=frames):
            buf = _io.StringIO()
            _write_simple_xyz(buf, fs)
        def rs_w(fs=frames):
            buf = _io.StringIO()
            _write_simple_xyz(buf, fs)

        r = _measure(f"xyz_write/{n_frames}fr_{n_at}at", py_w, rs_w, n_reps)
        results.append(r); _print_row(r)

        buf_data = _io.StringIO()
        _write_simple_xyz(buf_data, frames)
        xyz_text = buf_data.getvalue()

        def py_r(d=xyz_text): list(_read_simple_xyz(_io.StringIO(d), index=slice(None)))
        def rs_r(d=xyz_text): list(_read_simple_xyz(_io.StringIO(d), index=slice(None)))

        r = _measure(f"xyz_read/{n_frames}fr_{n_at}at", py_r, rs_r, n_reps)
        results.append(r); _print_row(r)

    # ── 9. Minkowski reduction ────────────────────────────────────────────────
    print("\n  Minkowski reduction")
    triclinic = np.array([[5.0, 2.3, 1.1], [0.0, 4.8, 1.9], [0.0, 0.0, 3.7]])
    fcc_cell  = bulk('Cu', 'fcc', a=3.615).cell.array

    for label, cell in [("triclinic", triclinic), ("fcc", fcc_cell)]:
        def py(c=cell): minkowski_reduce(c)
        def rs(c=cell): minkowski_reduce(c)
        r = _measure(f"minkowski/{label}", py, rs, n_reps)
        results.append(r); _print_row(r)

    return results


# ── formatting ────────────────────────────────────────────────────────────────

HEADER = (
    f"  {'Operation':<38s}  {'Baseline ms':>11s}  {'ase-fast ms':>11s}  {'Speedup':>7s}"
)
SEP = "  " + "-" * 74


def _print_row(r):
    spd = f"{r['speedup_x']:.1f}×"
    print(
        f"  {r['label']:<38s}  {r['py_ms']:>11.2f}  {r['rs_ms']:>11.2f}  {spd:>7s}"
    )


def _to_markdown(results, meta):
    lines = [
        "## ASE-fast benchmark — baseline vs Rust",
        "",
        f"**Platform:** {meta['platform']}  ",
        f"**Python:** {meta['python']}  ",
        f"**Date:** {meta['timestamp']}  ",
        "",
        "| Operation | Baseline (ms) | ase-fast (ms) | Speedup |",
        "|-----------|:-------------:|:-------------:|:-------:|",
    ]
    for r in results:
        spd = f"**{r['speedup_x']:.1f}×**" if r['speedup_x'] >= 2 else f"{r['speedup_x']:.1f}×"
        lines.append(
            f"| `{r['label']}` | {r['py_ms']:.2f} | {r['rs_ms']:.2f} | {spd} |"
        )
    lines += [
        "",
        "> Measured with `python benchmarks/compare.py`. "
        "Speedup = baseline / ase-fast median wall-clock time (higher is better).",
    ]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    # Check Rust is active
    flags = {
        "neighborlist": _nl._HAVE_RUST_NEIGHBORLIST,
        "extxyz":       _extxyz._HAVE_RUST_EXTXYZ,
        "geometry":     _mink._HAVE_RUST_GEOM,
        "io":           _vasp._HAVE_RUST_IO,
    }
    print("\n" + "=" * 74)
    print("  ASE-fast — speed comparison (baseline Python vs Rust)")
    for k, v in flags.items():
        status = "ACTIVE ✓" if v else "NOT FOUND ✗"
        print(f"    {k:<16s} {status}")
    if not all(flags.values()):
        print("\n  WARNING: some Rust extensions missing — run `pip install -e .` first")
    print("=" * 74)
    print(HEADER)
    print(SEP)

    results = run_all(n_reps=args.reps)

    # Summary
    print("\n" + SEP)
    with_speedup = [r for r in results if r["speedup_x"] >= 1.0]
    avg_speed = sum(r["speedup_x"] for r in with_speedup) / len(with_speedup)
    best      = max(with_speedup, key=lambda r: r["speedup_x"])
    print(f"\n  Average speedup across all operations: {avg_speed:.1f}×")
    print(f"  Best speedup:   {best['speedup_x']:.1f}×  ({best['label']})")

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "platform":  platform.platform(),
        "python":    sys.version.split()[0],
        "rust_flags": flags,
    }

    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    json_path = Path(args.output) if args.output else out_dir / f"{ts}_compare.json"
    md_path   = json_path.with_suffix(".md")

    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    with open(md_path, "w") as f:
        f.write(_to_markdown(results, meta))

    print(f"\n  JSON → {json_path}")
    print(f"  MD   → {md_path}  (paste into README)\n")


if __name__ == "__main__":
    main()
