"""Extended Phase-4.5 tests for ase.io.extxyz.

Covers:
  - key_val_str_to_dict, key_val_dict_to_str (task 5)
  - parse_properties, read_xyz, write_xyz, save_calc_results (task 6)
"""

import io
import textwrap

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.build import bulk, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.extxyz import (
    key_val_dict_to_str,
    key_val_str_to_dict,
    parse_properties,
    read_xyz,
    save_calc_results,
    write_xyz,
)


# ===========================================================================
# Task 5 — key_val_str_to_dict and key_val_dict_to_str
# ===========================================================================

class TestKeyValStrToDict:

    def test_empty_string(self):
        # ASE's parser maps '' to {'': True}; confirm it returns a dict
        result = key_val_str_to_dict('')
        assert isinstance(result, dict)

    def test_single_int(self):
        result = key_val_str_to_dict('n=5')
        assert result['n'] == 5
        assert isinstance(result['n'], (int, np.integer))

    def test_single_float(self):
        result = key_val_str_to_dict('e=-1.23')
        assert np.isclose(result['e'], -1.23)

    def test_bool_true(self):
        result = key_val_str_to_dict('pbc=T')
        assert result['pbc'] is True

    def test_bool_false(self):
        result = key_val_str_to_dict('pbc=F')
        assert result['pbc'] is False

    def test_bool_True_False_words(self):
        result = key_val_str_to_dict('a=True b=False')
        assert result['a'] is True
        assert result['b'] is False

    def test_quoted_string_with_space(self):
        result = key_val_str_to_dict('label="hello world"')
        assert result['label'] == 'hello world'

    def test_unquoted_string(self):
        result = key_val_str_to_dict('sym=Cu')
        assert result['sym'] == 'Cu'

    def test_array_of_ints(self):
        result = key_val_str_to_dict('v={1 2 3}')
        assert list(result['v']) == [1, 2, 3]

    def test_3x3_lattice_matrix(self):
        lat = 'Lattice="3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0"'
        result = key_val_str_to_dict(lat)
        assert 'Lattice' in result
        assert result['Lattice'].shape == (3, 3)

    def test_mixed_types_in_one_string(self):
        s = 'energy=-1.5 natoms=2 periodic=T label="test mol"'
        result = key_val_str_to_dict(s)
        assert np.isclose(result['energy'], -1.5)
        assert result['natoms'] == 2
        assert result['periodic'] is True
        assert result['label'] == 'test mol'

    def test_regression_five_known_inputs(self):
        """Regression: freeze output for known comment-line inputs."""
        cases = [
            ('energy=0.0', {'energy': 0.0}),
            ('n=42', {'n': 42}),
            ('flag=T', {'flag': True}),
            ('flag=F', {'flag': False}),
            ('s="abc"', {'s': 'abc'}),
        ]
        for s, expected in cases:
            result = key_val_str_to_dict(s)
            for k, v in expected.items():
                assert k in result
                if isinstance(v, float):
                    assert np.isclose(result[k], v)
                else:
                    assert result[k] == v, f"For input '{s}': {result[k]!r} != {v!r}"


class TestKeyValDictToStr:

    def test_round_trip_str_dict_str(self):
        s = 'energy=-1.5 natoms=2'
        d = key_val_str_to_dict(s)
        s2 = key_val_dict_to_str(d)
        d2 = key_val_str_to_dict(s2)
        for k in d:
            if isinstance(d[k], (int, float)):
                assert np.isclose(d2[k], d[k])
            else:
                assert d2[k] == d[k]

    def test_bool_true_becomes_T(self):
        s = key_val_dict_to_str({'flag': True})
        assert 'flag=T' in s

    def test_bool_false_becomes_F(self):
        s = key_val_dict_to_str({'flag': False})
        assert 'flag=F' in s

    def test_numpy_array_space_separated(self):
        d = {'v': np.array([1.0, 2.0, 3.0])}
        s = key_val_dict_to_str(d)
        # should contain three numbers
        assert '1' in s and '2' in s and '3' in s


# ===========================================================================
# Task 6 — parse_properties, read_xyz, write_xyz, save_calc_results
# ===========================================================================

class TestParseProperties:

    def test_standard_species_pos(self):
        prop_str = 'species:S:1:pos:R:3'
        props, props_list, dtype, converters = parse_properties(prop_str)
        assert 'species' in props
        assert 'pos' in props

    def test_with_forces_column(self):
        prop_str = 'species:S:1:pos:R:3:forces:R:3'
        props, props_list, dtype, converters = parse_properties(prop_str)
        assert 'forces' in props

    def test_logical_column(self):
        prop_str = 'species:S:1:pos:R:3:mask:L:1'
        props, props_list, dtype, converters = parse_properties(prop_str)
        assert 'mask' in props


def _make_extxyz_string(atoms, **info):
    """Helper: write atoms to in-memory extxyz string."""
    buf = io.StringIO()
    write_xyz(buf, atoms, **info)
    buf.seek(0)
    return buf


def _atoms_equal(a1, a2, atol=1e-8):
    """Check two Atoms objects have the same content."""
    if len(a1) != len(a2):
        return False
    if not np.allclose(a1.positions, a2.positions, atol=atol):
        return False
    if list(a1.symbols) != list(a2.symbols):
        return False
    if not np.allclose(a1.cell[:], a2.cell[:], atol=atol):
        return False
    if not np.array_equal(a1.pbc, a2.pbc):
        return False
    return True


class TestReadWriteXyz:

    def test_single_frame_round_trip(self):
        atoms = bulk('Cu', 'fcc', a=3.615) * 2
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        assert _atoms_equal(atoms, atoms2)

    def test_multiframe_trajectory_order(self):
        frames = [molecule('H2O') for _ in range(5)]
        for i, f in enumerate(frames):
            f.info['frame_idx'] = i
        buf = io.StringIO()
        write_xyz(buf, frames)
        buf.seek(0)
        read_back = list(read_xyz(buf, index=slice(None)))
        assert len(read_back) == 5
        for i, f in enumerate(read_back):
            assert f.info['frame_idx'] == i

    def test_index_minus1_returns_last_frame(self):
        frames = [molecule('H2') for _ in range(3)]
        for i, f in enumerate(frames):
            f.info['idx'] = i
        buf = io.StringIO()
        write_xyz(buf, frames)
        buf.seek(0)
        last = next(read_xyz(buf, index=-1))
        assert last.info['idx'] == 2

    def test_index_0_returns_first_frame(self):
        frames = [molecule('H2') for _ in range(3)]
        for i, f in enumerate(frames):
            f.info['idx'] = i
        buf = io.StringIO()
        write_xyz(buf, frames)
        buf.seek(0)
        first = next(read_xyz(buf, index=0))
        assert first.info['idx'] == 0

    def test_per_atom_arrays_preserved(self):
        # per-atom arrays named 'forces' are read back via SinglePointCalculator
        atoms = bulk('Cu', 'fcc', a=3.615) * 2
        forces = np.random.default_rng(0).random((len(atoms), 3))
        atoms.arrays['forces'] = forces
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        # forces land in the attached calculator results
        assert atoms2.calc is not None
        assert 'forces' in atoms2.calc.results
        assert np.allclose(atoms2.calc.results['forces'], forces, atol=1e-8)

    def test_per_config_info_preserved(self):
        atoms = bulk('Al', 'fcc', a=4.05)
        atoms.info['energy'] = -3.72
        atoms.info['label'] = 'test'
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        # label stays in info; energy is absorbed into the calculator
        assert atoms2.info['label'] == 'test'
        assert atoms2.calc is not None and np.isclose(
            atoms2.calc.results['energy'], -3.72
        )

    def test_stress_tensor_round_trip(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        stress = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        atoms.info['stress'] = stress
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        # stress is absorbed into the calculator results
        assert atoms2.calc is not None and 'stress' in atoms2.calc.results
        assert np.allclose(atoms2.calc.results['stress'], stress, atol=1e-8)

    def test_zero_atom_frame(self):
        atoms = Atoms()
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        assert len(atoms2) == 0

    def test_non_orthorhombic_cell_preserved(self):
        a = 3.57
        b = a / 2
        cell = [[0, b, b], [b, 0, b], [b, b, 0]]
        atoms = Atoms('C', positions=[[0, 0, 0]], cell=cell, pbc=True)
        buf = _make_extxyz_string(atoms)
        atoms2 = next(read_xyz(buf, index=0))
        assert np.allclose(atoms2.cell[:], atoms.cell[:], atol=1e-8)

    def test_plain_true_produces_standard_xyz(self):
        atoms = molecule('H2O')
        buf = io.StringIO()
        write_xyz(buf, atoms, plain=True)
        buf.seek(0)
        content = buf.read()
        # plain XYZ: line 2 is a blank comment, no key=val pairs
        lines = content.strip().split('\n')
        assert len(lines[1].strip()) == 0 or '=' not in lines[1]

    def test_large_trajectory_no_error(self):
        """100 frames — should write and read back without error."""
        atoms = molecule('H2O')  # 3-atom molecule
        buf = io.StringIO()
        write_xyz(buf, [atoms] * 100)
        buf.seek(0)
        frames = list(read_xyz(buf, index=slice(None)))
        assert len(frames) == 100


class TestSaveCalcResults:

    def _make_atoms_with_calc(self):
        atoms = bulk('Cu', 'fcc', a=3.615)
        calc = SinglePointCalculator(
            atoms,
            energy=-3.5,
            forces=np.zeros((len(atoms), 3)),
        )
        atoms.calc = calc
        return atoms

    def test_transfers_energy_to_info(self):
        atoms = self._make_atoms_with_calc()
        save_calc_results(atoms)
        # Default: keys are prefixed with the calculator class name
        all_keys = list(atoms.info.keys()) + list(atoms.arrays.keys())
        assert any('energy' in k for k in all_keys)

    def test_transfers_forces_to_arrays(self):
        atoms = self._make_atoms_with_calc()
        save_calc_results(atoms)
        all_keys = list(atoms.arrays.keys())
        assert any('forces' in k for k in all_keys)

    def test_calc_prefix_applied(self):
        atoms = self._make_atoms_with_calc()
        save_calc_results(atoms, calc_prefix='dft_')
        # at least one key should start with 'dft_'
        all_keys = list(atoms.info.keys()) + list(atoms.arrays.keys())
        assert any(k.startswith('dft_') for k in all_keys)

    def test_remove_atoms_calc_detaches_calc(self):
        atoms = self._make_atoms_with_calc()
        save_calc_results(atoms, remove_atoms_calc=True)
        assert atoms.calc is None

    def test_force_false_does_not_overwrite(self):
        atoms = self._make_atoms_with_calc()
        atoms.info['energy'] = 999.0
        save_calc_results(atoms, force=False)
        # should not overwrite
        assert atoms.info['energy'] == 999.0


# ===========================================================================
# Phase-8 Rust contracts — pin exact text format for drop-in verification
#
# These tests pin the exact text output that the Rust extxyz writer must
# reproduce.  If a test here fails after adding Rust, the Rust code is wrong.
# ===========================================================================

def _write_to_string(atoms, **kwargs) -> str:
    """Write atoms to extxyz string and return it."""
    buf = io.StringIO()
    write_xyz(buf, atoms, **kwargs)
    return buf.getvalue()


class TestExtxyzFormatContracts:
    """Exact output format contracts for Phase-8 Rust extxyz."""

    def test_first_line_is_natoms(self):
        """Line 1 must be the atom count as a plain integer."""
        atoms = molecule('H2O')
        text = _write_to_string(atoms)
        first_line = text.splitlines()[0].strip()
        assert first_line == str(len(atoms))

    def test_second_line_contains_lattice_key(self):
        """Line 2 (comment) must contain 'Lattice='."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        assert 'Lattice=' in comment

    def test_lattice_has_nine_floats(self):
        """Lattice= value must be 9 space-separated numbers (3×3 row-major)."""
        atoms = bulk('Al', 'fcc', a=4.05, cubic=True)
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        # Extract Lattice= value (between quotes)
        import re
        m = re.search(r'Lattice="([^"]+)"', comment)
        assert m is not None, "Lattice= not found or not quoted"
        vals = m.group(1).split()
        assert len(vals) == 9, f"Expected 9 lattice values, got {len(vals)}"
        # Must parse as floats
        floats = [float(v) for v in vals]
        assert len(floats) == 9

    def test_lattice_row_major_order(self):
        """Lattice values must be in row-major (C) order matching cell[:]."""
        a = 4.05
        atoms = bulk('Al', 'fcc', a=a, cubic=True)
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        import re
        m = re.search(r'Lattice="([^"]+)"', comment)
        vals = [float(v) for v in m.group(1).split()]
        cell_flat = atoms.cell[:].flatten().tolist()
        assert np.allclose(vals, cell_flat, atol=1e-6)

    def test_properties_string_present(self):
        """Comment line must contain 'Properties='."""
        atoms = molecule('H2O')
        text = _write_to_string(atoms)
        assert 'Properties=' in text.splitlines()[1]

    def test_properties_contains_species_and_pos(self):
        """Properties must include species:S:1 and pos:R:3."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        import re
        m = re.search(r'Properties=(\S+)', comment)
        props = m.group(1)
        assert 'species:S:1' in props
        assert 'pos:R:3' in props

    def test_pbc_encoding_true(self):
        """Periodic system: pbc must be encoded as 'T T T'."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        assert 'T T T' in comment or 'pbc="T T T"' in comment

    def test_pbc_encoding_false(self):
        """Non-periodic system: pbc must be encoded as 'F F F'."""
        atoms = molecule('H2O')
        text = _write_to_string(atoms)
        comment = text.splitlines()[1]
        assert 'F F F' in comment or 'pbc="F F F"' in comment

    def test_atom_line_count_equals_natoms(self):
        """Number of atom data lines must equal natoms."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        text = _write_to_string(atoms)
        lines = text.strip().splitlines()
        assert len(lines) == 2 + len(atoms)  # header + comment + atoms

    def test_integer_array_type_code(self):
        """Integer per-atom array must use type code 'I' in Properties."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        atoms.arrays['tag'] = np.arange(len(atoms), dtype=int)
        text = _write_to_string(atoms)
        assert 'tag:I:1' in text.splitlines()[1]

    def test_real_array_type_code(self):
        """Float per-atom array must use type code 'R' in Properties."""
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        atoms.arrays['charge'] = np.ones(len(atoms), dtype=float)
        text = _write_to_string(atoms)
        assert 'charge:R:1' in text.splitlines()[1]

    def test_multiframe_frame_boundaries(self):
        """Multi-frame file: each frame starts with its own atom count."""
        frames = [molecule('H2'), molecule('H2O')]
        text = _write_to_string(frames)
        lines = text.strip().splitlines()
        assert lines[0] == '2'   # H2 has 2 atoms
        # H2: 2 + 1(comment) + 2(atoms) = 4 lines; next frame starts at line 5
        assert lines[4] == '3'   # H2O has 3 atoms

    def test_float_precision_survives_round_trip(self):
        """Positions must survive write→read with at least 8 significant figures."""
        rng = np.random.default_rng(42)
        atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
        atoms.positions += rng.uniform(-0.1, 0.1, atoms.positions.shape)
        orig_pos = atoms.positions.copy()
        buf = io.StringIO(_write_to_string(atoms))
        atoms2 = next(read_xyz(buf, index=0))
        assert np.allclose(atoms2.positions, orig_pos, atol=1e-6)
