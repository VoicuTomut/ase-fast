"""Extra tests for RFO"""

import numpy as np
import pytest  # noqa: F401
import scipy

from ase.calculators.emt import EMT
from ase.optimize import RFO


def test_rfo_eigh_fallback(distorted_bulk_gold, monkeypatch):
    """Test eigh fallback when eigsh fails to converge"""
    e1, n1 = run_emt_rfo(distorted_bulk_gold.copy())
    monkeypatch.setattr(scipy.sparse.linalg, 'eigsh', bad_eigsh)
    e2, n2 = run_emt_rfo(distorted_bulk_gold.copy())
    assert e1 == pytest.approx(e2, abs=1e-6)
    assert n1 == n2


def run_emt_rfo(atoms):
    """Return final energy and number of steps for optimizing
    atoms with EMT and RFO.
    """
    atoms.calc = EMT()
    opt = RFO(atoms)
    opt.run()
    return atoms.get_potential_energy(), opt.nsteps


def bad_eigsh(*args, **kwargs):
    """Fake eigsh that raises convergence failure."""
    raise scipy.sparse.linalg.ArpackNoConvergence(
        'No convergence', np.array([]), np.array([])
    )
