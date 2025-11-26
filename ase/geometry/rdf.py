# fmt: off

import math
from typing import List, Optional, Tuple, Union

import numpy as np

from ase import Atoms
from ase.cell import Cell
from ase.neighborlist import NeighborList


class CellTooSmall(Exception):
    pass


class VolumeNotDefined(Exception):
    pass


def get_rdf(atoms: Atoms, rmax: float, nbins: int,
            distance_matrix: Optional[np.ndarray] = None,
            elements: Optional[Union[List[int], Tuple]] = None,
            no_dists: Optional[bool] = False,
            volume: Optional[float] = None):
    """Returns two numpy arrays; the radial distribution function
    and the corresponding distances of the supplied atoms object.
    If no_dists = True then only the first array is returned.

    Note that the rdf is computed following the standard solid state
    definition which uses the cell volume in the normalization.
    This may or may not be appropriate in cases where one or more
    directions is non-periodic.

    Parameters:

    rmax : float
        The maximum distance that will contribute to the rdf.
        The unit cell should be large enough so that it encloses a
        sphere with radius rmax in the periodic directions.

    nbins : int
        Number of bins to divide the rdf into.

    distance_matrix : numpy.array
        An array of distances between atoms, typically
        obtained by atoms.get_all_distances().
        Default None meaning that a NeighborList will be used.

    elements : list or tuple
        List of two atomic numbers. If elements is not None the partial
        rdf for the supplied elements will be returned.

    no_dists : bool
        If True then the second array with rdf distances will not be returned.

    volume : float or None
        Optionally specify the volume of the system. If specified, the volume
        will be used instead atoms.cell.volume.
    """

    # First check whether the cell is sufficiently large
    vol = atoms.cell.volume if volume is None else volume
    if vol < 1.0e-10:
        raise VolumeNotDefined

    check_cell_and_r_max(atoms, rmax)

    natoms = len(atoms)
    rdf = np.zeros(nbins + 1)
    dr = float(rmax / nbins)

    if elements is None:
        i_indices = np.arange(natoms)
    else:
        i_indices = np.where(atoms.numbers == elements[0])[0]

    phi = len(i_indices) / vol
    norm = 2.0 * math.pi * dr * phi * natoms

    if distance_matrix is None:
        nl = NeighborList(np.ones(natoms) * rmax * 0.5)
        nl.update(atoms)

        for i in i_indices:
            j_indices, _ = nl.get_neighbors(i)
            if elements is not None:
                j_indices = [
                        j for j in j_indices if atoms.numbers[j] == elements[1]
                ]

            if not len(j_indices):
                continue

            distances = atoms.get_distances(i, j_indices, mic=True)
            indices = np.asarray(np.ceil(distances / dr), dtype=int)
            for index in range(nbins + 1):
                rdf[index] += np.count_nonzero(indices == index)
    else:
        indices = np.asarray(np.ceil(distance_matrix / dr), dtype=int)
        if elements is None:
            indices_triu = np.triu(indices)
            for index in range(nbins + 1):
                rdf[index] = np.count_nonzero(indices_triu == index)

        else:
            norm *= 2
            for i in i_indices:
                for j in np.where(atoms.numbers == elements[1])[0]:
                    index = indices[i, j]
                    if index <= nbins:
                        rdf[index] += 1

    rr = np.arange(dr / 2, rmax, dr)
    rdf[1:] /= norm * (rr * rr + (dr * dr / 12))

    if no_dists:
        return rdf[1:]

    return rdf[1:], rr


def check_cell_and_r_max(atoms: Atoms, rmax: float) -> None:
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    vol = atoms.cell.volume

    for i in range(3):
        if pbc[i]:
            axb = np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :])
            h = vol / np.linalg.norm(axb)
            if h < 2 * rmax:
                recommended_r_max = get_recommended_r_max(cell, pbc)
                raise CellTooSmall(
                    'The cell is not large enough in '
                    f'direction {i}: {h:.3f} < 2*rmax={2 * rmax: .3f}. '
                    f'Recommended rmax = {recommended_r_max}')


def get_recommended_r_max(cell: Cell, pbc: List[bool]) -> float:
    recommended_r_max = 5.0
    vol = cell.volume
    for i in range(3):
        if pbc[i]:
            axb = np.cross(cell[(i + 1) % 3, :],  # type: ignore[index]
                           cell[(i + 2) % 3, :])  # type: ignore[index]
            h = vol / np.linalg.norm(axb)
            assert isinstance(h, float)  # mypy
            recommended_r_max = min(h / 2 * 0.99, recommended_r_max)
    return recommended_r_max


def get_containing_cell_length(atoms: Atoms) -> np.ndarray:
    atom2xyz = atoms.get_positions()
    return np.amax(atom2xyz, axis=0) - np.amin(atom2xyz, axis=0) + 2.0


def get_volume_estimate(atoms: Atoms) -> float:
    return np.prod(get_containing_cell_length(atoms))
