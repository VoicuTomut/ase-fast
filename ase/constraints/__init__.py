"""Constraints"""

from copy import deepcopy
from typing import Any

from ase.constraints.constraint import (
    FixConstraint,
    constrained_indices,
)
from ase.constraints.externalforce import ExternalForce
from ase.constraints.fixatoms import FixAtoms
from ase.constraints.fixbondlengths import FixBondLength, FixBondLengths
from ase.constraints.fixcartesian import FixCartesian
from ase.constraints.fixcom import FixCom, FixSubsetCom
from ase.constraints.fixedline import FixedLine
from ase.constraints.fixedmode import FixedMode
from ase.constraints.fixedplane import FixedPlane
from ase.constraints.fixinternals import FixInternals
from ase.constraints.fixlineartriatomic import FixLinearTriatomic
from ase.constraints.fixparametricrelations import (
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
)
from ase.constraints.fixscaled import FixScaled
from ase.constraints.fixsymmetry import FixSymmetry
from ase.constraints.hookean import Hookean
from ase.constraints.mirrorforce import MirrorForce
from ase.constraints.mirrortorque import MirrorTorque

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
