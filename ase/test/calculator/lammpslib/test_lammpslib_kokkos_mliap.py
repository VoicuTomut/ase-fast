# fmt: off
import pytest

from ase import Atom, Atoms
from ase.build import bulk


@pytest.fixture()
def atoms_fcc_Ni_with_H_at_center():
    atoms = bulk("Ni", cubic=True)
    atoms += Atom("H", position=atoms.cell.diagonal() / 2)
    return atoms


@pytest.fixture()
def atoms_fcc_Ar():
    atoms = bulk("Ar", a=1.0, cubic=True)
    return atoms


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammpslib_simple_kokkos(
    factory,
    calc_params_NiH: dict,
    calc_params_kokkos_cpu: dict,
    atoms_fcc_Ni_with_H_at_center: Atoms,
):
    NiH = atoms_fcc_Ni_with_H_at_center

    # Add a bit of distortion to the cell
    NiH.set_cell(
        NiH.cell + [[0.1, 0.2, 0.4], [0.3, 0.2, 0.0], [0.1, 0.1, 0.1]],
        scale_atoms=True,
    )

    calc_params = calc_params_NiH.copy()
    calc_params.update(calc_params_kokkos_cpu)
    calc = factory.calc(**calc_params)
    NiH.calc = calc

    _ = NiH.get_potential_energy()


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammpslib_simple_mliap(
    factory,
    calc_params_Ar_mliap: dict,
    atoms_fcc_Ar: Atoms,
):
    Ar_bulk = atoms_fcc_Ar

    # Add a bit of distortion to the cell
    Ar_bulk.set_cell(
        Ar_bulk.cell + [[0.1, 0.2, 0.4], [0.3, 0.2, 0.0], [0.1, 0.1, 0.1]],
        scale_atoms=True,
    )

    calc_params = calc_params_Ar_mliap.copy()
    calc = factory.calc(**calc_params)
    Ar_bulk.calc = calc

    _ = Ar_bulk.get_potential_energy()
