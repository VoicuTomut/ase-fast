import numpy as np
import pytest

import ase.lattice as lattice


@pytest.mark.parametrize('noise', [0, 1e-6, 1e-3, 1e-1])
def test_match_to_lattice(noise: float):
    rng = np.random.RandomState(42)

    lat = lattice.ORCI(2.1, 3.2, 4.3)
    cell = lat.tocell() + noise * rng.random((3, 3))

    matches = [*lattice.match_to_lattice(cell, 'ORCI')]
    assert len(matches) == 1
    match = matches[0]

    tolerance = 2 * noise + 1e-12
    assert match.error < tolerance
    assert match.lat.name == 'ORCI'
    assert lattice.celldiff(match.lat.tocell(), lat.tocell()) < tolerance
