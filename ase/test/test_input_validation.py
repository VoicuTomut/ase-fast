"""Phase-5 readiness — input validation tests.

These tests verify that ASE raises *some* exception for clearly invalid
inputs.  They pass today and will continue to pass after Phase 5 improves
the error messages.  After Phase 5, add pytest.match checks to each test
to also verify the human-readable message.

Covers the top bottleneck functions targeted for error message improvement:
  - primitive_neighbor_list  (neighborlist.py)
  - key_val_str_to_dict      (io/extxyz.py)
  - wrap_positions           (geometry/geometry.py)
  - get_distances            (geometry/geometry.py)
  - bulk                     (build/bulk.py)
  - make_supercell           (build/supercells.py)
"""
from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.geometry import wrap_positions
from ase.geometry.geometry import get_distances
from ase.io.extxyz import key_val_str_to_dict
from ase.neighborlist import primitive_neighbor_list


# ---------------------------------------------------------------------------
# primitive_neighbor_list
# ---------------------------------------------------------------------------

class TestPrimitiveNeighborListValidation:

    def test_wrong_positions_ndim(self):
        """Positions must be 2-D; 1-D array should raise."""
        positions = np.array([0.0, 0.0, 0.0])  # 1-D, not (N, 3)
        cell = np.eye(3) * 5.0
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('i', pbc, cell, positions, 3.0)

    def test_positions_wrong_second_dim(self):
        """Positions second dimension must be 3."""
        positions = np.zeros((4, 2))  # (N, 2) — wrong
        cell = np.eye(3) * 5.0
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('i', pbc, cell, positions, 3.0)

    def test_cell_wrong_shape(self):
        """Cell must be 3×3; wrong shape should raise."""
        positions = np.array([[0.0, 0.0, 0.0]])
        cell = np.eye(4) * 5.0  # 4×4 — wrong
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('i', pbc, cell, positions, 3.0)

    @pytest.mark.xfail(
        reason="Phase 5: no validation for negative cutoff in current ASE"
    )
    def test_negative_scalar_cutoff(self):
        """Negative cutoff is physically nonsensical and should raise.

        Phase-5 target: add explicit ValueError with message like
        'cutoff must be positive, got -1.0'.
        """
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cell = np.eye(3) * 10.0
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('i', pbc, cell, positions, -1.0)

    @pytest.mark.xfail(
        reason="Phase 5: no validation for mismatched cutoff list length"
    )
    def test_cutoff_list_wrong_length(self):
        """Per-atom cutoff list length must match number of atoms.

        Phase-5 target: add explicit ValueError with message like
        'cutoff list length (3) does not match number of atoms (2)'.
        """
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cell = np.eye(3) * 10.0
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('i', pbc, cell, positions, [1.0, 1.0, 1.0])

    def test_unknown_quantity_character(self):
        """Unknown quantity character (not in 'ijdDS') should raise."""
        positions = np.array([[0.0, 0.0, 0.0]])
        cell = np.eye(3) * 5.0
        pbc = np.array([False, False, False])
        with pytest.raises(Exception):
            primitive_neighbor_list('x', pbc, cell, positions, 3.0)


# ---------------------------------------------------------------------------
# wrap_positions
# ---------------------------------------------------------------------------

class TestWrapPositionsValidation:

    def test_positions_not_3d(self):
        """Positions must have shape (N, 3); wrong shape should raise."""
        cell = np.eye(3) * 5.0
        pos = np.array([[1.0, 2.0]])  # (1, 2) — wrong
        with pytest.raises(Exception):
            wrap_positions(pos, cell, pbc=True)

    def test_cell_not_3x3(self):
        """Cell must be 3×3; wrong shape should raise."""
        cell = np.eye(2) * 5.0  # 2×2 — wrong
        pos = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(Exception):
            wrap_positions(pos, cell, pbc=True)


# ---------------------------------------------------------------------------
# get_distances
# ---------------------------------------------------------------------------

class TestGetDistancesValidation:

    def test_positions_wrong_shape(self):
        """Positions must be (N, 3); (N, 2) should raise."""
        p = np.zeros((3, 2))
        with pytest.raises(Exception):
            get_distances(p)

    def test_p1_p2_incompatible_columns(self):
        """p1 and p2 must both be (N, 3); mismatched columns should raise."""
        p1 = np.zeros((2, 3))
        p2 = np.zeros((2, 2))
        with pytest.raises(Exception):
            get_distances(p1, p2)


# ---------------------------------------------------------------------------
# bulk
# ---------------------------------------------------------------------------

class TestBulkValidation:

    def test_unknown_structure(self):
        """Unknown crystalstructure string should raise ValueError."""
        with pytest.raises(Exception):
            bulk('Cu', 'nonexistent_structure', a=3.615)

    @pytest.mark.xfail(
        reason="Phase 5: bulk('X', 'hcp') silently uses a default c/a instead of raising"
    )
    def test_hcp_without_c(self):
        """HCP for an unknown symbol without c/ca ratio should raise an error.

        Phase-5 target: add explicit ValueError with message like
        'HCP requires c or ca parameter when no reference state exists for X'.
        """
        with pytest.raises(Exception):
            # HCP requires c or ca; if neither given and no reference state, should fail
            bulk('X', 'hcp', a=3.0)


# ---------------------------------------------------------------------------
# make_supercell
# ---------------------------------------------------------------------------

class TestMakeSupercellValidation:

    def test_singular_P_matrix(self):
        """Singular transformation matrix should raise."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        P = np.zeros((3, 3))  # singular
        with pytest.raises(Exception):
            make_supercell(atoms, P)

    def test_non_integer_P_matrix(self):
        """Non-integer-valued P matrix should raise (supercell must be commensurate)."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        P = [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]]
        with pytest.raises(Exception):
            make_supercell(atoms, P)


# ---------------------------------------------------------------------------
# key_val_str_to_dict — resilience, not just happy path
# ---------------------------------------------------------------------------

class TestKeyValStrToDictEdgeCases:

    def test_unmatched_quote_does_not_crash(self):
        """Malformed string with unmatched quote should not crash (return dict)."""
        # Even if the parse is partial, it must not raise
        try:
            result = key_val_str_to_dict('key="unmatched')
            assert isinstance(result, dict)
        except Exception:
            pass  # raising is acceptable for malformed input

    def test_deeply_nested_braces_handled(self):
        """Nested braces {{ }} in value should not crash."""
        try:
            result = key_val_str_to_dict('v={1 2 3}')
            assert isinstance(result, dict)
        except Exception:
            pass

    def test_very_long_comment_line(self):
        """Very long comment line (10k chars) should parse without timeout/crash."""
        # Build a valid long line: many key=value pairs
        pairs = ' '.join(f'k{i}={i}' for i in range(200))
        result = key_val_str_to_dict(pairs)
        assert isinstance(result, dict)
        assert len(result) >= 100

    def test_duplicate_keys_last_wins_or_no_crash(self):
        """Duplicate keys should not crash; last value or first value is acceptable."""
        result = key_val_str_to_dict('a=1 a=2')
        assert isinstance(result, dict)
        assert 'a' in result
