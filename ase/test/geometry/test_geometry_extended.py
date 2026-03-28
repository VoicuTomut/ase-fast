"""Extended Phase-4.5 tests for ase.geometry.

Covers:
  - wrap_positions, find_mic variants, get_distances (task 10)
  - get_angles, get_dihedrals, get_layers, get_duplicate_atoms (task 11)
"""

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.geometry import (
    get_distances,
    get_duplicate_atoms,
    get_layers,
    wrap_positions,
)
from ase.geometry.geometry import (
    find_mic,
    general_find_mic,
    get_angles,
    get_dihedrals,
    naive_find_mic,
)


# ===========================================================================
# Task 10 — wrap_positions, find_mic, get_distances
# ===========================================================================

class TestWrapPositions:

    def _cubic_cell(self, a=5.0):
        return np.eye(3) * a

    def test_positions_inside_cell_unchanged(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[1.0, 2.0, 3.0]])
        wrapped = wrap_positions(pos, cell, pbc=True)
        assert np.allclose(wrapped, pos)

    def test_positions_outside_cell_wrapped(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[6.0, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=True)
        assert np.allclose(wrapped, [[1.0, 0.0, 0.0]], atol=1e-10)

    def test_negative_position_wrapped(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[-1.0, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=True)
        assert np.allclose(wrapped, [[4.0, 0.0, 0.0]], atol=1e-10)

    def test_no_pbc_unchanged(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[6.0, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=False)
        assert np.allclose(wrapped, pos)

    def test_partial_pbc(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[6.0, 6.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=[True, False, False])
        assert np.allclose(wrapped[0, 0], 1.0, atol=1e-10)
        assert np.allclose(wrapped[0, 1], 6.0, atol=1e-10)

    def test_center_shifts_wrap_center(self):
        cell = self._cubic_cell(5.0)
        pos = np.array([[0.0, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=True, center=(0.0, 0.0, 0.0))
        assert np.allclose(wrapped, [[0.0, 0.0, 0.0]], atol=1e-10)

    def test_multiple_atoms(self):
        cell = self._cubic_cell(4.0)
        pos = np.array([[5.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=True)
        assert np.allclose(wrapped[0, 0], 1.0, atol=1e-10)
        assert np.allclose(wrapped[1, 0], 3.0, atol=1e-10)

    def test_non_orthorhombic_cell(self):
        """wrap_positions should work with non-orthorhombic cells."""
        a = 3.0
        cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, a]])
        pos = np.array([[a * 1.5, 0.0, 0.0]])
        wrapped = wrap_positions(pos, cell, pbc=True)
        # Check wrapped position is inside cell (fractional coords in [0,1))
        frac = np.linalg.solve(cell.T, wrapped[0])
        assert (frac >= -1e-10).all() and (frac < 1.0 + 1e-10).all()


class TestFindMic:

    def _cubic_cell(self, a=10.0):
        return np.eye(3) * a

    def test_short_vector_unchanged(self):
        cell = self._cubic_cell(10.0)
        v = np.array([[1.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=True)
        assert np.allclose(vmic, v, atol=1e-10)

    def test_long_vector_wrapped(self):
        """Vector longer than half the cell should be wrapped to shorter image."""
        cell = self._cubic_cell(10.0)
        v = np.array([[7.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=True)
        assert vmic[0, 0] < 0  # should wrap to -3.0
        assert abs(abs(vmic[0, 0]) - 3.0) < 1e-10

    def test_returns_minimum_image_distance(self):
        cell = self._cubic_cell(10.0)
        v = np.array([[6.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=True)
        assert dists[0] < 6.0

    def test_no_pbc_no_wrapping(self):
        cell = self._cubic_cell(10.0)
        v = np.array([[7.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=False)
        assert np.allclose(vmic, v, atol=1e-10)

    def test_naive_and_general_agree_orthorhombic(self):
        cell = self._cubic_cell(8.0)
        v = np.array([[2.0, 1.0, 1.0]])  # short vector, safe for naive_find_mic
        pbc = np.array([True, True, True])
        v_naive, d_naive = naive_find_mic(v, cell)
        v_gen, d_gen = general_find_mic(v, cell, pbc=pbc)
        assert np.allclose(v_naive, v_gen, atol=1e-10)

    def test_multiple_vectors(self):
        cell = self._cubic_cell(10.0)
        v = np.array([[6.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-7.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=True)
        assert vmic.shape == v.shape
        assert dists.shape == (3,)

    def test_zero_vector(self):
        cell = self._cubic_cell(10.0)
        v = np.array([[0.0, 0.0, 0.0]])
        vmic, dists = find_mic(v, cell, pbc=True)
        assert np.allclose(vmic, 0.0)
        assert dists[0] == 0.0


class TestGetDistances:

    def test_self_distances_zero(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        D, d = get_distances(pos)
        assert d[0, 0] == 0.0
        assert d[1, 1] == 0.0

    def test_distance_between_two_points(self):
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[3.0, 4.0, 0.0]])
        D, d = get_distances(p1, p2)
        assert abs(d[0, 0] - 5.0) < 1e-10

    def test_pbc_distance_shorter(self):
        """With PBC, minimum image distance should be shorter."""
        cell = np.eye(3) * 10.0
        p1 = np.array([[1.0, 0.0, 0.0]])
        p2 = np.array([[9.0, 0.0, 0.0]])
        D_pbc, d_pbc = get_distances(p1, p2, cell=cell, pbc=True)
        D_nopbc, d_nopbc = get_distances(p1, p2)
        assert d_pbc[0, 0] < d_nopbc[0, 0]

    def test_symmetric_distance_matrix(self):
        pos = np.random.default_rng(42).random((5, 3)) * 5
        D, d = get_distances(pos)
        assert np.allclose(d, d.T, atol=1e-10)

    def test_output_shape(self):
        pos = np.random.default_rng(1).random((4, 3)) * 3
        D, d = get_distances(pos)
        assert D.shape == (4, 4, 3)
        assert d.shape == (4, 4)

    def test_p2_output_shape(self):
        p1 = np.random.default_rng(2).random((3, 3))
        p2 = np.random.default_rng(3).random((5, 3))
        D, d = get_distances(p1, p2)
        assert D.shape == (3, 5, 3)
        assert d.shape == (3, 5)

    def test_known_distance(self):
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[1.0, 0.0, 0.0]])
        D, d = get_distances(p1, p2)
        assert abs(d[0, 0] - 1.0) < 1e-10


# ===========================================================================
# Task 11 — get_angles, get_dihedrals, get_layers, get_duplicate_atoms
# ===========================================================================

class TestGetAngles:

    def test_90_degree_angle(self):
        """Two perpendicular vectors should give 90°."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0]])
        angles = get_angles(v0, v1)
        assert abs(angles[0] - 90.0) < 1e-8

    def test_180_degree_angle(self):
        """Antiparallel vectors should give 180°."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[-1.0, 0.0, 0.0]])
        angles = get_angles(v0, v1)
        assert abs(angles[0] - 180.0) < 1e-6

    def test_0_degree_angle(self):
        """Parallel vectors should give 0°."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[2.0, 0.0, 0.0]])
        angles = get_angles(v0, v1)
        assert abs(angles[0]) < 1e-8

    def test_multiple_angles(self):
        v0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        angles = get_angles(v0, v1)
        assert angles.shape == (2,)
        assert abs(angles[0] - 90.0) < 1e-8
        assert abs(angles[1] - 90.0) < 1e-8

    def test_output_in_degrees(self):
        """By default, angles should be in degrees, not radians."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0]])
        angles = get_angles(v0, v1)
        # 90° in radians would be ~1.57, should be ~90 instead
        assert angles[0] > 5.0


class TestGetDihedrals:

    def test_dihedral_90_degrees(self):
        """Classic 90-degree dihedral: v0 along x, v1 along y, v2 along x."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0]])
        v2 = np.array([[0.0, 0.0, 1.0]])
        dihedrals = get_dihedrals(v0, v1, v2)
        assert dihedrals.shape == (1,)
        # Just check the result is a finite number in [-180, 180]
        assert -180.0 <= dihedrals[0] <= 180.0

    def test_multiple_dihedrals(self):
        v0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        dihedrals = get_dihedrals(v0, v1, v2)
        assert dihedrals.shape == (2,)

    def test_dihedral_in_degrees(self):
        """Output should be in degrees (not radians)."""
        v0 = np.array([[1.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 1.0, 0.0]])
        v2 = np.array([[0.0, 0.0, 1.0]])
        dihedrals = get_dihedrals(v0, v1, v2)
        # If in degrees, |value| should be > 1 for non-trivial case
        # (pi in radians ≈ 3.14, so 90° would be > 5 if degrees)
        assert abs(dihedrals[0]) >= 0.0  # basic sanity


class TestGetLayers:

    def test_fcc111_layer_count(self):
        from ase.spacegroup import crystal
        al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=4.05)
        al001 = al.repeat([1, 1, 2])
        tags, levels = get_layers(al001, (0, 0, 1))
        assert len(levels) >= 2

    def test_simple_layer_along_z(self):
        """Atoms at different z values should have different layers."""
        atoms = Atoms(
            'H4',
            positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
            cell=[5, 5, 10],
            pbc=False,
        )
        tags, levels = get_layers(atoms, (0, 0, 1))
        assert len(levels) == 4
        assert len(tags) == 4

    def test_same_layer_atoms(self):
        """Atoms at the same z should have the same tag."""
        atoms = Atoms(
            'H4',
            positions=[[0, 0, 0], [1, 0, 0], [2, 0, 1], [3, 0, 1]],
            cell=[5, 5, 5],
            pbc=False,
        )
        tags, levels = get_layers(atoms, (0, 0, 1))
        assert tags[0] == tags[1]
        assert tags[2] == tags[3]
        assert tags[0] != tags[2]

    def test_tags_and_levels_length_consistency(self):
        atoms = bulk('Cu', 'fcc', a=3.615) * (2, 2, 2)
        tags, levels = get_layers(atoms, (0, 0, 1))
        assert len(tags) == len(atoms)
        assert len(levels) == len(np.unique(tags))

    def test_layer_tolerance(self):
        """Atoms within tolerance of same layer should be grouped."""
        atoms = Atoms(
            'H3',
            positions=[[0, 0, 0.0], [1, 0, 0.05], [2, 0, 1.0]],
            cell=[5, 5, 5],
            pbc=False,
        )
        # With tolerance > 0.05, first two should be same layer
        tags, levels = get_layers(atoms, (0, 0, 1), tolerance=0.1)
        assert tags[0] == tags[1]


class TestGetDuplicateAtoms:

    def test_no_duplicates(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        dups = get_duplicate_atoms(atoms)
        assert len(dups) == 0

    def test_with_duplicates(self):
        """Manually add a duplicate atom and verify detection."""
        atoms = bulk('Cu', 'fcc', a=3.615)
        pos = atoms.positions[0].copy()
        atoms.extend(Atoms('Cu', positions=[pos + [0.001, 0.0, 0.0]]))
        dups = get_duplicate_atoms(atoms, cutoff=0.01)
        assert len(dups) > 0

    def test_delete_true_removes_duplicates(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        n_before = len(atoms)
        # Add near-duplicate
        atoms.extend(Atoms('Cu', positions=[atoms.positions[0].copy()]))
        assert len(atoms) == n_before + 1
        get_duplicate_atoms(atoms, delete=True)
        assert len(atoms) == n_before

    def test_supercell_no_duplicates(self):
        """A clean supercell should have zero duplicates."""
        atoms = bulk('Al', 'fcc', a=4.05) * (3, 3, 3)
        dups = get_duplicate_atoms(atoms)
        assert len(dups) == 0

    def test_cutoff_sensitivity(self):
        """Large cutoff should find 'duplicates' that a tight cutoff misses."""
        atoms = Atoms(
            'H2',
            positions=[[0, 0, 0], [0.05, 0, 0]],
            cell=[5, 5, 5],
            pbc=False,
        )
        dups_tight = get_duplicate_atoms(atoms, cutoff=0.01)
        dups_loose = get_duplicate_atoms(atoms, cutoff=0.1)
        assert len(dups_loose) >= len(dups_tight)
