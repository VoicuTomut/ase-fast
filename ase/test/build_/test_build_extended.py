"""Extended Phase-4.5 tests for ase.build.

Covers:
  - bulk() crystal structures (task 7)
  - make_supercell, find_optimal_cell_shape, lattice_points_in_supercell (task 8)
  - surface functions and add_adsorbate/add_vacuum (task 9)
"""

import numpy as np
import pytest

from ase import Atoms
from ase.build import (
    add_adsorbate,
    add_vacuum,
    bcc100,
    bcc110,
    bcc111,
    bulk,
    diamond100,
    diamond111,
    fcc100,
    fcc110,
    fcc111,
    fcc211,
    hcp0001,
    make_supercell,
)
from ase.build.supercells import (
    eval_length_deviation,
    find_optimal_cell_shape,
    lattice_points_in_supercell,
)


# ===========================================================================
# Task 7 — bulk() crystal structures
# ===========================================================================

class TestBulkCrystalStructures:

    def test_fcc_cu_default(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        assert len(atoms) == 1
        assert 'Cu' in atoms.get_chemical_formula()
        assert atoms.pbc.all()

    def test_bcc_fe_default(self):
        atoms = bulk('Fe', 'bcc', a=2.87)
        assert len(atoms) == 1
        assert 'Fe' in atoms.get_chemical_formula()

    def test_hcp_ti(self):
        atoms = bulk('Ti', 'hcp', a=2.95, c=4.68)
        assert len(atoms) == 2

    def test_diamond_si(self):
        atoms = bulk('Si', 'diamond', a=5.43)
        assert len(atoms) == 2

    def test_sc_simple_cubic(self):
        atoms = bulk('Po', 'sc', a=3.0)
        assert len(atoms) == 1

    def test_orthorhombic_fcc(self):
        atoms = bulk('Cu', 'fcc', a=3.615, orthorhombic=True)
        lattice = atoms.cell.get_bravais_lattice()
        angles = lattice.cellpar()[3:]
        assert abs(angles - 90.0).max() < 1e-8

    def test_cubic_fcc(self):
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        lattice = atoms.cell.get_bravais_lattice()
        assert lattice.name == 'CUB'

    def test_cubic_bcc(self):
        atoms = bulk('Fe', 'bcc', a=2.87, cubic=True)
        lattice = atoms.cell.get_bravais_lattice()
        assert lattice.name == 'CUB'

    def test_cubic_diamond(self):
        atoms = bulk('Si', 'diamond', a=5.43, cubic=True)
        lattice = atoms.cell.get_bravais_lattice()
        assert lattice.name == 'CUB'

    def test_fcc_scaled_positions_sum_to_one(self):
        """All scaled positions of FCC unit cell should be in [0, 1)."""
        atoms = bulk('Cu', 'fcc', a=3.615)
        spos = atoms.get_scaled_positions()
        assert ((spos >= 0.0) & (spos < 1.0)).all()

    def test_cell_volume_positive(self):
        atoms = bulk('Al', 'fcc', a=4.05)
        assert atoms.get_volume() > 0.0

    def test_cell_volume_scales_with_a(self):
        """Volume should scale as a^3 for cubic structures."""
        a1, a2 = 3.0, 6.0
        v1 = bulk('Cu', 'fcc', a=a1).get_volume()
        v2 = bulk('Cu', 'fcc', a=a2).get_volume()
        assert abs(v2 / v1 - (a2 / a1) ** 3) < 1e-8

    def test_pbc_all_true_for_periodic_bulk(self):
        atoms = bulk('Au', 'fcc', a=4.08)
        assert atoms.pbc.all()

    def test_supercell_multiplied(self):
        unit = bulk('Cu', 'fcc', a=3.615)
        supercell = unit * (2, 2, 2)
        assert len(supercell) == 8 * len(unit)

    def test_rocksalt_nacl(self):
        atoms = bulk('NaCl', 'rocksalt', a=5.64)
        assert len(atoms) == 2
        symbols = set(atoms.get_chemical_symbols())
        assert 'Na' in symbols and 'Cl' in symbols

    def test_zincblende_zns(self):
        atoms = bulk('ZnS', 'zincblende', a=5.41)
        assert len(atoms) == 2
        symbols = set(atoms.get_chemical_symbols())
        assert 'Zn' in symbols and 'S' in symbols

    def test_cesiumchloride_cscl(self):
        atoms = bulk('CsCl', 'cesiumchloride', a=4.12)
        assert len(atoms) == 2
        symbols = set(atoms.get_chemical_symbols())
        assert 'Cs' in symbols and 'Cl' in symbols

    def test_fluorite_cao2(self):
        atoms = bulk('CaF2', 'fluorite', a=5.46)
        assert len(atoms) == 3
        symbols = set(atoms.get_chemical_symbols())
        assert 'Ca' in symbols and 'F' in symbols


# ===========================================================================
# Task 8 — make_supercell, find_optimal_cell_shape, lattice_points_in_supercell
# ===========================================================================

class TestMakeSupercell:

    def test_diagonal_supercell(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        sc = make_supercell(atoms, P)
        assert len(sc) == 8 * len(atoms)

    def test_non_diagonal_supercell_atom_count(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        P = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        sc = make_supercell(atoms, P)
        # |det(P)| = 2, so 2 * len(atoms) atoms
        det = int(round(abs(np.linalg.det(P))))
        assert len(sc) == det * len(atoms)

    def test_pbc_preserved(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        P = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        sc = make_supercell(atoms, P)
        assert sc.pbc.all()

    def test_chemical_formula_preserved(self):
        atoms = bulk('NaCl', 'rocksalt', a=5.64)
        P = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        sc = make_supercell(atoms, P)
        # Supercell should still have equal numbers of Na and Cl
        syms = sc.get_chemical_symbols()
        assert syms.count('Na') == syms.count('Cl')

    def test_cell_volume_scales_with_det(self):
        atoms = bulk('Al', 'fcc', a=4.05)
        P = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        sc = make_supercell(atoms, P)
        det = int(round(abs(np.linalg.det(P))))
        assert abs(sc.get_volume() / atoms.get_volume() - det) < 1e-6


class TestFindOptimalCellShape:

    def test_target_size_8_fcc(self):
        """For 8-atom supercell of FCC, optimal shape should be near-cubic."""
        atoms = bulk('Cu', 'fcc', a=3.615)
        P = find_optimal_cell_shape(atoms.cell, 8, 'sc')
        assert P.shape == (3, 3)
        det = round(abs(np.linalg.det(P)))
        assert det == 8

    def test_returns_integer_matrix(self):
        atoms = bulk('Al', 'fcc', a=4.05)
        P = find_optimal_cell_shape(atoms.cell, 4, 'sc')
        assert P.dtype in (np.int32, np.int64, int) or np.issubdtype(P.dtype, np.integer)

    def test_deviation_improves_with_optimal(self):
        """Optimal shape should have lower deviation than identity scaled."""
        atoms = bulk('Cu', 'fcc', a=3.615)
        P_opt = find_optimal_cell_shape(atoms.cell, 4, 'sc')
        dev_opt = eval_length_deviation(atoms.cell @ P_opt, 'sc')
        # Identity * 4 is a valid but non-optimal supercell diagonal
        P_naive = np.diag([4, 1, 1])
        dev_naive = eval_length_deviation(atoms.cell @ P_naive, 'sc')
        assert dev_opt <= dev_naive


class TestLatticePointsInSupercell:

    def test_2x2x2_count(self):
        P = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        pts = lattice_points_in_supercell(P)
        assert len(pts) == 8

    def test_identity_gives_one_point(self):
        P = np.eye(3, dtype=int)
        pts = lattice_points_in_supercell(P)
        assert len(pts) == 1

    def test_points_in_unit_cell(self):
        """All fractional coords of returned lattice points should be in [0,1)."""
        P = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        pts = lattice_points_in_supercell(P)
        assert len(pts) == 27
        # points are in fractional coords of P, so check in [0,1)
        assert ((pts >= 0.0 - 1e-10) & (pts < 1.0 + 1e-10)).all()


# ===========================================================================
# Task 9 — surface functions and add_adsorbate/add_vacuum
# ===========================================================================

class TestSurfaceFunctions:

    def test_fcc111_layers_and_size(self):
        slab = fcc111('Al', size=(1, 1, 4), a=4.05)
        assert len(slab) == 4

    def test_fcc100_layers(self):
        slab = fcc100('Cu', size=(1, 1, 3), a=3.615)
        assert len(slab) == 3

    def test_fcc110_layers(self):
        slab = fcc110('Cu', size=(1, 1, 4), a=3.615)
        assert len(slab) == 4

    def test_bcc100_layers(self):
        slab = bcc100('Fe', size=(1, 1, 4), a=2.87)
        assert len(slab) == 4

    def test_bcc110_layers(self):
        slab = bcc110('Fe', size=(1, 1, 4), a=2.87)
        assert len(slab) == 4

    def test_bcc111_layers(self):
        slab = bcc111('Fe', size=(1, 1, 4), a=2.87)
        assert len(slab) == 4

    def test_diamond100_layers(self):
        slab = diamond100('Si', size=(1, 1, 4), a=5.43)
        assert len(slab) == 4

    def test_diamond111_layers(self):
        slab = diamond111('C', size=(1, 1, 4), a=3.57)
        assert len(slab) == 4

    def test_hcp0001_layers(self):
        slab = hcp0001('Ti', size=(1, 1, 4), a=2.95, c=4.68)
        assert len(slab) == 4

    def test_fcc211_layers(self):
        slab = fcc211('Cu', size=(3, 1, 4), a=3.615)
        assert len(slab) == 3 * 1 * 4

    def test_surface_pbc(self):
        """Slab should be periodic in x/y but not necessarily z."""
        slab = fcc111('Al', size=(2, 2, 3), a=4.05)
        # x and y should be periodic
        assert slab.pbc[0] and slab.pbc[1]

    def test_surface_cell_c_direction(self):
        """z-cell length should accommodate layers plus vacuum."""
        slab = fcc111('Al', size=(1, 1, 4), a=4.05, vacuum=5.0)
        # Cell z-length must be positive and larger than atom z-span
        z_extent = slab.positions[:, 2].max() - slab.positions[:, 2].min()
        cell_z = np.linalg.norm(slab.cell[2])
        assert cell_z > z_extent


class TestAddAdsorbate:

    def test_add_one_adsorbate(self):
        slab = fcc111('Al', size=(2, 2, 3), a=4.05, vacuum=5.0)
        n_before = len(slab)
        add_adsorbate(slab, 'H', height=1.5, position='ontop')
        assert len(slab) == n_before + 1

    def test_add_molecule_adsorbate(self):
        from ase.build import molecule
        slab = fcc111('Cu', size=(2, 2, 3), a=3.615, vacuum=5.0)
        n_before = len(slab)
        co = molecule('CO')
        add_adsorbate(slab, co, height=2.0, position='ontop')
        assert len(slab) == n_before + 2  # CO has 2 atoms

    def test_adsorbate_above_surface(self):
        slab = fcc111('Al', size=(2, 2, 3), a=4.05, vacuum=5.0)
        z_max_before = slab.positions[:, 2].max()
        add_adsorbate(slab, 'H', height=1.5, position='ontop')
        z_h = slab.positions[-1, 2]
        assert z_h > z_max_before


class TestAddVacuum:

    def test_add_vacuum_increases_cell_c(self):
        # Use vacuum=0.1 so the cell z-vector is non-zero (avoids div-by-zero)
        slab = fcc111('Al', size=(1, 1, 3), a=4.05, vacuum=0.1)
        c_before = slab.cell[2, 2]
        add_vacuum(slab, 10.0)
        c_after = slab.cell[2, 2]
        assert abs(c_after - c_before - 10.0) < 1e-8

    def test_add_vacuum_does_not_move_atoms(self):
        slab = fcc111('Al', size=(1, 1, 3), a=4.05, vacuum=0.1)
        positions_before = slab.positions.copy()
        add_vacuum(slab, 8.0)
        assert np.allclose(slab.positions, positions_before)

    def test_double_vacuum_doubled_cell(self):
        slab = fcc111('Al', size=(1, 1, 3), a=4.05, vacuum=0.1)
        add_vacuum(slab, 5.0)
        c1 = slab.cell[2, 2]
        add_vacuum(slab, 5.0)
        c2 = slab.cell[2, 2]
        assert abs(c2 - c1 - 5.0) < 1e-8
