import pytest

from ase.build import bulk


@pytest.fixture()
def distorted_bulk_gold():
    """Return 2x2x2 bulk gold supercell with first atom distorted in x"""
    atoms = bulk('Au')
    atoms *= (2, 2, 2)
    atoms[0].x += 0.5
    return atoms
