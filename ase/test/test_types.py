"""Phase-4.5 mypy CI gate — task 12.

Runs mypy in a subprocess on all files annotated during Phases 1–4
and asserts zero errors. Skipped if mypy is not installed.
"""

import subprocess
import sys

import pytest

# Files annotated during Phases 1–4 (relative to the ase package root).
# Add new paths here as more phases annotate files.
ANNOTATED_FILES = [
    "ase/atoms.py",
    "ase/atom.py",
    "ase/symbols.py",
    "ase/calculators/calculator.py",
    "ase/calculators/singlepoint.py",
    "ase/io/__init__.py",
    "ase/io/formats.py",
    "ase/io/extxyz.py",
    "ase/build/surface.py",
    "ase/neighborlist.py",
    "ase/geometry/geometry.py",
]

MYPY_FLAGS = [
    "--ignore-missing-imports",
    "--no-error-summary",
    "--no-strict-optional",
]


def _mypy_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _mypy_available(), reason="mypy not installed")
@pytest.mark.parametrize("filepath", ANNOTATED_FILES)
def test_mypy_no_errors(filepath, tmp_path):
    """mypy should report zero errors for each annotated file."""
    cmd = [
        sys.executable, "-m", "mypy",
        *MYPY_FLAGS,
        filepath,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(
            # Run from the repo root so ase package is importable
            __import__("pathlib").Path(__file__).parents[3]
        ),
    )
    errors = [
        line for line in result.stdout.splitlines()
        if ": error:" in line
    ]
    assert errors == [], (
        f"mypy reported errors in {filepath}:\n" + "\n".join(errors)
    )
