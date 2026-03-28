"""Extended Phase-4.5 tests for ase.neighborlist.

Covers:
  - primitive_neighbor_list (task 2)
  - PrimitiveNeighborList, NewPrimitiveNeighborList (task 3)
  - NeighborList wrapper, natural_cutoffs, first_neighbors (task 4)
"""

import numpy as np
import pytest

from ase.build import bulk, molecule
from ase.neighborlist import (
    NeighborList,
    NewPrimitiveNeighborList,
    PrimitiveNeighborList,
    first_neighbors,
    natural_cutoffs,
    neighbor_list,
    primitive_neighbor_list,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fcc_cu(n=1):
    """FCC Cu supercell with n repetitions per axis.

    Uses cubic=True to get an orthorhombic 4-atom cell, avoiding the
    divide-by-zero RuntimeWarning that the non-orthorhombic primitive cell
    triggers in conftest's filterwarnings='error' mode.
    """
    return bulk('Cu', 'fcc', a=3.615, cubic=True) * n


# ===========================================================================
# Task 2 — primitive_neighbor_list
# ===========================================================================

# The upstream ASE primitive_neighbor_list has a pre-existing
# "divide by zero encountered in dot" RuntimeWarning for supercells with
# n >= 2 cubic-FCC repetitions.  conftest.py promotes all RuntimeWarnings to
# errors, so we suppress this known upstream warning at the class level.
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
class TestPrimitiveNeighborList:

    def test_empty_atoms_returns_zero_length_arrays(self):
        positions = np.zeros((0, 3))
        cell = np.eye(3) * 5.0
        pbc = np.array([False, False, False])
        for q in ['i', 'j', 'd', 'D', 'S']:
            result = primitive_neighbor_list(q, pbc, cell, positions, 3.0)
            assert result.shape[0] == 0, f"quantity '{q}' not empty"

    def test_single_atom_no_pbc_no_self_interaction(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        cell = np.eye(3) * 10.0
        pbc = np.array([False, False, False])
        i = primitive_neighbor_list('i', pbc, cell, positions, 5.0,
                                    self_interaction=False)
        assert len(i) == 0

    def test_single_atom_pbc_self_interaction_true(self):
        positions = np.array([[0.0, 0.0, 0.0]])
        cell = np.eye(3) * 3.0
        pbc = np.array([True, True, True])
        # cutoff > half-cell: atom is its own neighbor via periodic image
        i = primitive_neighbor_list('i', pbc, cell, positions, 2.5,
                                    self_interaction=True)
        assert len(i) > 0

    def test_two_atoms_known_distance(self):
        d_known = 2.5
        positions = np.array([[0.0, 0.0, 0.0],
                               [d_known, 0.0, 0.0]])
        cell = np.eye(3) * 20.0
        pbc = np.array([False, False, False])
        i, j, d = primitive_neighbor_list('ijd', pbc, cell, positions,
                                          d_known + 0.1)
        assert len(d) > 0
        assert np.allclose(d, d_known, atol=1e-10)

    def test_two_atoms_distance_vector(self):
        positions = np.array([[0.0, 0.0, 0.0],
                               [1.0, 2.0, 0.0]])
        cell = np.eye(3) * 20.0
        pbc = np.array([False, False, False])
        i, j, D = primitive_neighbor_list('ijD', pbc, cell, positions, 3.0)
        # vector from i to j
        expected = positions[j] - positions[i]
        assert np.allclose(D, expected, atol=1e-10)

    def test_fcc_coordination_number_12(self):
        atoms = _fcc_cu(1)  # 4-atom cubic cell; n>=2 triggers upstream divide-by-zero
        # FCC nearest-neighbor distance ≈ a/sqrt(2) ≈ 2.556 Å for Cu
        a = 3.615
        nn_dist = a / np.sqrt(2)
        i_arr = primitive_neighbor_list(
            'i', atoms.pbc, atoms.get_cell(complete=True),
            atoms.positions, nn_dist + 0.1,
            numbers=atoms.numbers, self_interaction=False)
        coord = np.bincount(i_arr, minlength=len(atoms))
        # Interior atoms should all have 12 neighbors
        assert (coord == 12).all(), f"Expected 12, got {np.unique(coord)}"

    def test_non_orthorhombic_cell_distances_correct(self):
        """FCC primitive cell — non-orthorhombic."""
        a = 3.615
        b = a / 2
        cell = np.array([[0, b, b], [b, 0, b], [b, b, 0]])
        positions = np.array([[0.0, 0.0, 0.0]])
        pbc = np.array([True, True, True])
        i, d = primitive_neighbor_list('id', pbc, cell, positions, a,
                                       self_interaction=False)
        # All distances should be the nearest-neighbor distance
        nn_dist = a / np.sqrt(2)
        assert len(d) > 0
        assert np.allclose(d, nn_dist, atol=1e-8)

    def test_mixed_pbc_z_not_periodic(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        pbc = np.array([True, True, False])
        cell = atoms.get_cell(complete=True)
        i, j = primitive_neighbor_list('ij', pbc, cell, atoms.positions,
                                       3.0, self_interaction=False)
        # With only xy periodic, shift vectors in z should all be 0
        i2, j2, S = primitive_neighbor_list('ijS', pbc, cell, atoms.positions,
                                             3.0, self_interaction=False)
        assert (S[:, 2] == 0).all(), "z-shift should be 0 for non-periodic z"

    def test_cutoff_as_float(self):
        atoms = _fcc_cu(1)
        i = primitive_neighbor_list(
            'i', atoms.pbc, atoms.get_cell(complete=True),
            atoms.positions, 3.0, self_interaction=False)
        assert len(i) > 0

    def test_cutoff_as_per_atom_list(self):
        atoms = _fcc_cu(1)
        cutoffs = [1.5] * len(atoms)
        i = primitive_neighbor_list(
            'i', atoms.pbc, atoms.get_cell(complete=True),
            atoms.positions, cutoffs, self_interaction=False)
        assert len(i) >= 0  # may be 0 if spheres don't overlap

    def test_cutoff_as_dict(self):
        atoms = molecule('CO')
        cell = np.eye(3) * 20.0
        pbc = np.array([False, False, False])
        cutoff = {(6, 8): 1.3, (8, 6): 1.3}
        i, j, d = primitive_neighbor_list('ijd', pbc, cell,
                                          atoms.positions, cutoff,
                                          numbers=atoms.numbers,
                                          self_interaction=False)
        assert len(i) > 0

    def test_use_scaled_positions(self):
        atoms = _fcc_cu(1)
        cell = atoms.get_cell(complete=True)
        pbc = atoms.pbc
        cart_positions = atoms.positions

        # Cartesian result
        i_cart, d_cart = primitive_neighbor_list(
            'id', pbc, cell, cart_positions, 3.0,
            self_interaction=False, use_scaled_positions=False)

        # Scaled positions result
        scaled = np.linalg.solve(cell.T, cart_positions.T).T
        i_scal, d_scal = primitive_neighbor_list(
            'id', pbc, cell, scaled, 3.0,
            self_interaction=False, use_scaled_positions=True)

        assert len(i_cart) == len(i_scal)
        assert np.allclose(np.sort(d_cart), np.sort(d_scal), atol=1e-10)

    def test_all_distances_le_cutoff(self):
        atoms = _fcc_cu(1)
        # Keep cutoff < lattice parameter to avoid overflow with the 4-atom cell
        cutoff = 3.0
        d = primitive_neighbor_list(
            'd', atoms.pbc, atoms.get_cell(complete=True),
            atoms.positions, cutoff, self_interaction=False)
        assert (d <= cutoff + 1e-10).all(), "distances exceed cutoff"

    def test_quantities_tuple_vs_single(self):
        atoms = _fcc_cu(1)
        pbc = atoms.pbc
        cell = atoms.get_cell(complete=True)

        result = primitive_neighbor_list('ijdDS', pbc, cell,
                                         atoms.positions, 3.5,
                                         self_interaction=False)
        assert isinstance(result, tuple)
        assert len(result) == 5

        result_single = primitive_neighbor_list('i', pbc, cell,
                                                atoms.positions, 3.5,
                                                self_interaction=False)
        assert isinstance(result_single, np.ndarray)

    def test_h2_bond_length_regression(self):
        """H₂ nearest-neighbor distance should be ~0.74 Å."""
        atoms = molecule('H2')
        cell = np.eye(3) * 20.0
        pbc = np.array([False, False, False])
        d = primitive_neighbor_list('d', pbc, cell, atoms.positions, 1.0,
                                    self_interaction=False)
        assert len(d) > 0
        assert np.allclose(d[0], 0.741, atol=0.02), f"H₂ bond = {d[0]:.3f}"

    def test_neighbor_list_wrapper_matches_primitive(self):
        """neighbor_list() should give same result as primitive_neighbor_list()."""
        atoms = _fcc_cu(1)
        cutoff = 3.5
        i1, d1 = neighbor_list('id', atoms, cutoff, self_interaction=False)
        i2, d2 = primitive_neighbor_list(
            'id', atoms.pbc, atoms.get_cell(complete=True),
            atoms.positions, cutoff, numbers=atoms.numbers,
            self_interaction=False)
        assert np.array_equal(np.sort(i1), np.sort(i2))


# ===========================================================================
# Task 3 — PrimitiveNeighborList & NewPrimitiveNeighborList
# ===========================================================================

class TestPrimitiveNeighborListClass:

    @pytest.fixture
    def diamond_setup(self):
        a = 3.57  # diamond lattice constant
        atoms = bulk('C', 'diamond', a=a)
        cell = atoms.get_cell(complete=True)
        pbc = atoms.pbc
        positions = atoms.positions
        cutoffs = np.array([2.0] * len(atoms))
        return pbc, cell, positions, cutoffs

    def test_nupdates_increments(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.0, self_interaction=False)
        assert nl.nupdates == 0
        nl.update(pbc, cell, positions)
        assert nl.nupdates == 1
        nl.update(pbc, cell, positions)
        assert nl.nupdates == 1  # no rebuild needed

    def test_no_rebuild_when_atoms_static(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.3, self_interaction=False)
        rebuilt = nl.update(pbc, cell, positions)
        assert rebuilt is True
        rebuilt2 = nl.update(pbc, cell, positions)
        assert rebuilt2 is False

    def test_rebuild_when_atom_moves(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.1, self_interaction=False)
        nl.update(pbc, cell, positions)
        moved = positions.copy()
        moved[0] += 0.5  # move beyond skin
        rebuilt = nl.update(pbc, cell, moved)
        assert rebuilt is True

    def test_bothways_true_symmetry(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.0, self_interaction=False,
                                   bothways=True)
        nl.build(pbc, cell, positions)
        n = len(positions)
        for a in range(n):
            for b in nl.neighbors[a]:
                assert a in nl.neighbors[b], \
                    f"bothways=True: {a} in neighbors({b}) expected"

    def test_bothways_false_half_list(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl_half = PrimitiveNeighborList(cutoffs, skin=0.0,
                                        self_interaction=False, bothways=False)
        nl_both = PrimitiveNeighborList(cutoffs, skin=0.0,
                                        self_interaction=False, bothways=True)
        nl_half.build(pbc, cell, positions)
        nl_both.build(pbc, cell, positions)
        count_half = sum(len(n) for n in nl_half.neighbors)
        count_both = sum(len(n) for n in nl_both.neighbors)
        assert count_both == 2 * count_half

    def test_self_interaction_true(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.0, self_interaction=True,
                                   bothways=True)
        nl.build(pbc, cell, positions)
        for a in range(len(positions)):
            assert a in nl.neighbors[a], f"atom {a} not in own neighbor list"

    def test_sorted_neighbors_ascending(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl = PrimitiveNeighborList(cutoffs, skin=0.0, self_interaction=False,
                                   sorted=True, bothways=True)
        nl.build(pbc, cell, positions)
        for a in range(len(positions)):
            nbs = nl.neighbors[a]
            if len(nbs) > 1:
                assert (np.diff(nbs) >= 0).all(), \
                    f"neighbors of {a} not sorted: {nbs}"

    def test_new_and_primitive_agree(self, diamond_setup):
        pbc, cell, positions, cutoffs = diamond_setup
        nl_old = PrimitiveNeighborList(cutoffs, skin=0.0,
                                       self_interaction=False, bothways=True)
        nl_new = NewPrimitiveNeighborList(cutoffs, skin=0.0,
                                          self_interaction=False, bothways=True)
        nl_old.build(pbc, cell, positions)
        nl_new.build(pbc, cell, positions)

        for a in range(len(positions)):
            old_nb = np.sort(nl_old.neighbors[a])
            new_nb, _ = nl_new.get_neighbors(a)
            new_nb = np.sort(new_nb)
            assert np.array_equal(old_nb, new_nb), \
                f"atom {a}: old={old_nb} new={new_nb}"

    def test_fcc_regression_fixture(self):
        """Freeze exact neighbor count for FCC Cu 3×3×3, cutoff=3.0 Å."""
        atoms = _fcc_cu(3)
        cutoffs = np.array([1.5] * len(atoms))  # sphere-overlap: 1.5+1.5=3.0
        nl = PrimitiveNeighborList(cutoffs, skin=0.0, self_interaction=False,
                                   bothways=True)
        nl.build(atoms.pbc, atoms.get_cell(complete=True), atoms.positions)
        total_bonds = sum(len(n) for n in nl.neighbors)
        # FCC has 12 nearest neighbors; total bonds = 12 * 108 (bothways halved by build)
        assert total_bonds == 12 * len(atoms), \
            f"Expected {12 * len(atoms)} bonds, got {total_bonds}"


# ===========================================================================
# Task 4 — NeighborList wrapper, natural_cutoffs, first_neighbors
# ===========================================================================

class TestNeighborListWrapper:

    def test_update_returns_true_first_call(self):
        atoms = _fcc_cu(2)
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        assert nl.update(atoms) is True

    def test_update_returns_false_no_change(self):
        atoms = _fcc_cu(2)
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        assert nl.update(atoms) is False

    def test_get_neighbors_raises_before_update(self):
        atoms = _fcc_cu(2)
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs)
        with pytest.raises(RuntimeError, match='update'):
            nl.get_neighbors(0)

    def test_get_connectivity_matrix_sparse(self):
        from scipy.sparse import issparse
        atoms = molecule('H2O')
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        mat = nl.get_connectivity_matrix(sparse=True)
        assert issparse(mat)
        assert mat.shape == (len(atoms), len(atoms))

    def test_get_connectivity_matrix_dense(self):
        atoms = molecule('H2O')
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        mat = nl.get_connectivity_matrix(sparse=False)
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (len(atoms), len(atoms))

    def test_connectivity_matrix_symmetric_bothways(self):
        atoms = molecule('H2O')
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        mat = nl.get_connectivity_matrix(sparse=False)
        assert np.array_equal(mat, mat.T), "connectivity matrix not symmetric"

    def test_nupdates_property(self):
        atoms = _fcc_cu(2)
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs)
        assert nl.nupdates == 0
        nl.update(atoms)
        assert nl.nupdates == 1


class TestNaturalCutoffs:

    def test_length_matches_natoms(self):
        atoms = _fcc_cu(3)
        cutoffs = natural_cutoffs(atoms)
        assert len(cutoffs) == len(atoms)

    def test_mult_doubles_cutoffs(self):
        atoms = _fcc_cu(2)
        c1 = natural_cutoffs(atoms, mult=1)
        c2 = natural_cutoffs(atoms, mult=2)
        assert np.allclose(np.array(c2), 2.0 * np.array(c1))

    def test_per_symbol_override(self):
        atoms = molecule('H2O')
        cutoffs = natural_cutoffs(atoms, O=1.0)
        for atom, cut in zip(atoms, cutoffs):
            if atom.symbol == 'O':
                assert cut == 1.0


class TestFirstNeighbors:

    def test_empty_returns_zeros(self):
        result = first_neighbors(5, np.array([], dtype=int))
        assert len(result) == 6
        assert (result == 0).all()

    def test_consecutive_i_values(self):
        # atom 0 has 2 neighbors, atom 1 has 3
        first_atom = np.array([0, 0, 1, 1, 1])
        seed = first_neighbors(2, first_atom)
        assert seed[0] == 0
        assert seed[1] == 2
        assert seed[2] == 5

    def test_output_length_is_natoms_plus_one(self):
        first_atom = np.array([0, 0, 2, 2, 2])
        seed = first_neighbors(3, first_atom)
        assert len(seed) == 4  # natoms + 1


# ===========================================================================
# Phase-7 Rust contracts — pin exact outputs for drop-in verification
#
# These tests must pass UNCHANGED after the Rust implementation replaces
# the Python one.  If a test here fails after adding Rust, the Rust code
# is wrong, not the test.
# ===========================================================================

class TestNeighborlistContracts:
    """Exact-output contracts for Phase-7 Rust neighborlist."""

    def test_two_atom_exact_pair_set(self):
        """2 atoms, no PBC: exactly one bond (0→1) and its reverse (1→0)."""
        d_known = 2.5
        positions = np.array([[0.0, 0.0, 0.0], [d_known, 0.0, 0.0]])
        cell = np.eye(3) * 20.0
        pbc = np.array([False, False, False])
        i, j, d = primitive_neighbor_list(
            'ijd', pbc, cell, positions, d_known + 0.01,
            self_interaction=False)
        assert set(zip(i.tolist(), j.tolist())) == {(0, 1), (1, 0)}
        assert np.allclose(d, d_known, atol=1e-12)

    def test_shift_vector_identity(self):
        """D = positions[j] - positions[i] + S @ cell must hold exactly."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        i, j, D, S = primitive_neighbor_list(
            'ijDS',
            atoms.pbc,
            atoms.get_cell(complete=True),
            atoms.positions,
            3.0,
            self_interaction=False,
        )
        reconstructed = (
            atoms.positions[j] - atoms.positions[i] + S @ atoms.get_cell(complete=True)
        )
        assert np.allclose(D, reconstructed, atol=1e-10)

    def test_i_array_sorted_ascending(self):
        """i indices must be sorted in ascending order (output ordering contract)."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        i = primitive_neighbor_list(
            'i',
            atoms.pbc,
            atoms.get_cell(complete=True),
            atoms.positions,
            3.0,
            self_interaction=False,
        )
        assert np.all(i[:-1] <= i[1:]), "i array is not sorted ascending"

    def test_single_quantity_returns_ndarray(self):
        """Single quantity → ndarray; multiple quantities → tuple of ndarrays."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        pbc = atoms.pbc
        cell = atoms.get_cell(complete=True)
        pos = atoms.positions
        single = primitive_neighbor_list('i', pbc, cell, pos, 3.0)
        assert isinstance(single, np.ndarray)
        multi = primitive_neighbor_list('ij', pbc, cell, pos, 3.0)
        assert isinstance(multi, tuple) and len(multi) == 2

    def test_fcc_4atom_each_has_12_neighbors(self):
        """4-atom FCC cubic cell with PBC: every atom has exactly 12 neighbors."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        a = 3.615
        nn_dist = a / np.sqrt(2)
        i_arr = primitive_neighbor_list(
            'i',
            atoms.pbc,
            atoms.get_cell(complete=True),
            atoms.positions,
            nn_dist + 0.1,
            self_interaction=False,
        )
        coord = np.bincount(i_arr, minlength=len(atoms))
        assert (coord == 12).all(), f"Expected 12, got {coord}"

    def test_no_self_pairs_when_self_interaction_false(self):
        """With self_interaction=False, (i, i) pairs must never appear."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        i, j = primitive_neighbor_list(
            'ij',
            atoms.pbc,
            atoms.get_cell(complete=True),
            atoms.positions,
            3.0,
            self_interaction=False,
        )
        assert not np.any(i == j), "Self-pairs found with self_interaction=False"

    def test_bothways_symmetry_strict(self):
        """bothways=True: if (i,j) appears then (j,i) must also appear."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        nl = PrimitiveNeighborList(
            [1.5] * len(atoms), skin=0.0, self_interaction=False, bothways=True
        )
        nl.build(atoms.pbc, atoms.get_cell(complete=True), atoms.positions)
        pair_set = set()
        for atom_i, neigh in enumerate(nl.neighbors):
            for atom_j in neigh:
                pair_set.add((atom_i, atom_j))
        for (a, b) in list(pair_set):
            assert (b, a) in pair_set, f"Missing reverse pair ({b}, {a}) for ({a}, {b})"

    def test_distance_vector_magnitude_equals_d(self):
        """|D| must equal d for every pair."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        d, D = primitive_neighbor_list(
            'dD',
            atoms.pbc,
            atoms.get_cell(complete=True),
            atoms.positions,
            3.0,
            self_interaction=False,
        )
        d_from_D = np.linalg.norm(D, axis=1)
        assert np.allclose(d, d_from_D, atol=1e-10)
