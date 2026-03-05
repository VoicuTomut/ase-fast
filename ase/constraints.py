# fmt: off

"""Constraints"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence
from warnings import warn

import numpy as np

from ase import Atoms
from ase.geometry import (
    conditional_find_mic,
    find_mic,
    get_angles,
    get_angles_derivatives,
    get_dihedrals,
    get_dihedrals_derivatives,
    get_distances_derivatives,
    wrap_positions,
)
from ase.spacegroup.symmetrize import (
    prep_symmetry,
    refine_symmetry,
    symmetrize_rank1,
    symmetrize_rank2,
)
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from ase.utils.parsemath import eval_expression

__all__ = [
    'FixCartesian', 'FixBondLength', 'FixedMode',
    'FixAtoms', 'FixScaled', 'FixCom', 'FixSubsetCom', 'FixedPlane',
    'FixConstraint', 'FixedLine', 'FixBondLengths', 'FixLinearTriatomic',
    'FixInternals', 'Hookean', 'ExternalForce', 'MirrorForce', 'MirrorTorque',
    'FixScaledParametricRelations', 'FixCartesianParametricRelations',
    'FixSymmetry']


def dict2constraint(dct: dict[str, Any]) -> FixConstraint:
    """Convert dictionary to ASE `FixConstraint` object."""
    if dct['name'] not in __all__:
        raise ValueError
    # address backward-compatibility breaking between ASE 3.22.0 and 3.23.0
    # https://gitlab.com/ase/ase/-/merge_requests/3786
    if dct['name'] in {'FixedLine', 'FixedPlane'} and 'a' in dct['kwargs']:
        dct = deepcopy(dct)
        dct['kwargs']['indices'] = dct['kwargs'].pop('a')
    return globals()[dct['name']](**dct['kwargs'])


def slice2enlist(s, n):
    """Convert a slice object into a list of (new, old) tuples."""
    if isinstance(s, slice):
        return enumerate(range(*s.indices(n)))
    return enumerate(s)


def constrained_indices(atoms, only_include=None):
    """Returns a list of indices for the atoms that are constrained
    by a constraint that is applied.  By setting only_include to a
    specific type of constraint you can make it only look for that
    given constraint.
    """
    indices = []
    for constraint in atoms.constraints:
        if only_include is not None:
            if not isinstance(constraint, only_include):
                continue
        indices.extend(np.array(constraint.get_indices()))
    return np.array(np.unique(indices))


def ints2string(x, threshold=None):
    """Convert ndarray of ints to string."""
    if threshold is None or len(x) <= threshold:
        return str(x.tolist())
    return str(x[:threshold].tolist())[:-1] + ', ...]'


def _normalize(direction):
    if np.shape(direction) != (3,):
        raise ValueError("len(direction) is {len(direction)}. Has to be 3")

    direction = np.asarray(direction) / np.linalg.norm(direction)
    return direction


def _projection(vectors, direction):
    dotprods = vectors @ direction
    projection = direction[None, :] * dotprods[:, None]
    return projection


