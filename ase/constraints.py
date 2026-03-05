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


class FixConstraint:
    """Base class for classes that fix one or more atoms in some way."""

    def index_shuffle(self, atoms: Atoms, ind):
        """Change the indices.

        When the ordering of the atoms in the Atoms object changes,
        this method can be called to shuffle the indices of the
        constraints.

        ind -- List or tuple of indices.

        """
        raise NotImplementedError

    def repeat(self, m: int, n: int):
        """ basic method to multiply by m, needs to know the length
        of the underlying atoms object for the assignment of
        multiplied constraints to work.
        """
        msg = ("Repeat is not compatible with your atoms' constraints."
               ' Use atoms.set_constraint() before calling repeat to '
               'remove your constraints.')
        raise NotImplementedError(msg)

    def get_removed_dof(self, atoms: Atoms):
        """Get number of removed degrees of freedom due to constraint."""
        raise NotImplementedError

    def adjust_positions(self, atoms: Atoms, new):
        """Adjust positions."""

    def adjust_momenta(self, atoms: Atoms, momenta):
        """Adjust momenta."""
        # The default is in identical manner to forces.
        # TODO: The default is however not always reasonable.
        self.adjust_forces(atoms, momenta)

    def adjust_forces(self, atoms: Atoms, forces):
        """Adjust forces."""

    def copy(self):
        """Copy constraint."""
        return dict2constraint(self.todict().copy())

    def todict(self):
        """Convert constraint to dictionary."""


class IndexedConstraint(FixConstraint):
    def __init__(self, indices=None, mask=None):
        """Constrain chosen atoms.

        Parameters
        ----------
        indices : sequence of int
           Indices for those atoms that should be constrained.
        mask : sequence of bool
           One boolean per atom indicating if the atom should be
           constrained or not.
        """

        if mask is not None:
            if indices is not None:
                raise ValueError('Use only one of "indices" and "mask".')
            indices = mask
        indices = np.atleast_1d(indices)
        if np.ndim(indices) > 1:
            raise ValueError('indices has wrong amount of dimensions. '
                             f'Got {np.ndim(indices)}, expected ndim <= 1')

        if indices.dtype == bool:
            indices = np.arange(len(indices))[indices]
        elif len(indices) == 0:
            indices = np.empty(0, dtype=int)
        elif not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Indices must be integers or boolean mask, '
                             f'not dtype={indices.dtype}')

        if len(set(indices)) < len(indices):
            raise ValueError(
                'The indices array contains duplicates. '
                'Perhaps you want to specify a mask instead, but '
                'forgot the mask= keyword.')

        self.index = indices

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        index = []

        # Resolve negative indices:
        actual_indices = set(np.arange(len(atoms))[self.index])

        for new, old in slice2enlist(ind, len(atoms)):
            if old in actual_indices:
                index.append(new)
        if len(index) == 0:
            raise IndexError('All indices in FixAtoms not part of slice')
        self.index = np.asarray(index, int)
        # XXX make immutable

    def get_indices(self):
        return self.index.copy()

    def repeat(self, m, n):
        i0 = 0
        natoms = 0
        if isinstance(m, int):
            m = (m, m, m)
        index_new = []
        for _ in range(m[2]):
            for _ in range(m[1]):
                for _ in range(m[0]):
                    i1 = i0 + n
                    index_new += [i + natoms for i in self.index]
                    i0 = i1
                    natoms += n
        self.index = np.asarray(index_new, int)
        # XXX make immutable
        return self

    def delete_atoms(self, indices, natoms):
        """Removes atoms from the index array, if present.

        Required for removing atoms with existing constraint.
        """

        i = np.zeros(natoms, int) - 1
        new = np.delete(np.arange(natoms), indices)
        i[new] = np.arange(len(new))
        index = i[self.index]
        self.index = index[index >= 0]
        # XXX make immutable
        if len(self.index) == 0:
            return None
        return self


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


class FixSymmetry(FixConstraint):
    """
    Constraint to preserve spacegroup symmetry during optimisation.

    Requires spglib package to be available.
    """

    def __init__(self, atoms, symprec=0.01, adjust_positions=True,
                 adjust_cell=True, verbose=False):
        self.atoms = atoms.copy()
        self.symprec = symprec
        self.verbose = verbose
        refine_symmetry(atoms, symprec, self.verbose)  # refine initial symmetry
        sym = prep_symmetry(atoms, symprec, self.verbose)
        self.rotations, self.translations, self.symm_map = sym
        self.do_adjust_positions = adjust_positions
        self.do_adjust_cell = adjust_cell

    def adjust_cell(self, atoms, cell):
        if not self.do_adjust_cell:
            return
        # stress should definitely be symmetrized as a rank 2 tensor
        # UnitCellFilter uses deformation gradient as cell DOF with steps
        # dF = stress.F^-T quantity that should be symmetrized is therefore dF .
        # F^T assume prev F = I, so just symmetrize dF
        cur_cell = atoms.get_cell()
        cur_cell_inv = atoms.cell.reciprocal().T

        # F defined such that cell = cur_cell . F^T
        # assume prev F = I, so dF = F - I
        delta_deform_grad = np.dot(cur_cell_inv, cell).T - np.eye(3)

        # symmetrization doesn't work properly with large steps, since
        # it depends on current cell, and cell is being changed by deformation
        # gradient
        max_delta_deform_grad = np.max(np.abs(delta_deform_grad))
        if max_delta_deform_grad > 0.25:
            raise RuntimeError('FixSymmetry adjust_cell does not work properly'
                               ' with large deformation gradient step {} > 0.25'
                               .format(max_delta_deform_grad))
        elif max_delta_deform_grad > 0.15:
            warn('FixSymmetry adjust_cell may be ill behaved with large '
                 'deformation gradient step {}'.format(max_delta_deform_grad))

        symmetrized_delta_deform_grad = symmetrize_rank2(cur_cell, cur_cell_inv,
                                                         delta_deform_grad,
                                                         self.rotations)
        cell[:] = np.dot(cur_cell,
                         (symmetrized_delta_deform_grad + np.eye(3)).T)

    def adjust_positions(self, atoms, new):
        if not self.do_adjust_positions:
            return
        # symmetrize changes in position as rank 1 tensors
        step = new - atoms.positions
        symmetrized_step = symmetrize_rank1(atoms.get_cell(),
                                            atoms.cell.reciprocal().T, step,
                                            self.rotations, self.translations,
                                            self.symm_map)
        new[:] = atoms.positions + symmetrized_step

    def adjust_forces(self, atoms, forces):
        # symmetrize forces as rank 1 tensors
        # print('adjusting forces')
        forces[:] = symmetrize_rank1(atoms.get_cell(),
                                     atoms.cell.reciprocal().T, forces,
                                     self.rotations, self.translations,
                                     self.symm_map)

    def adjust_stress(self, atoms, stress):
        # symmetrize stress as rank 2 tensor
        raw_stress = voigt_6_to_full_3x3_stress(stress)
        symmetrized_stress = symmetrize_rank2(atoms.get_cell(),
                                              atoms.cell.reciprocal().T,
                                              raw_stress, self.rotations)
        stress[:] = full_3x3_to_voigt_6_stress(symmetrized_stress)

    def index_shuffle(self, atoms, ind):
        if len(atoms) != len(ind) or len(set(ind)) != len(ind):
            raise RuntimeError("FixSymmetry can only accomodate atom"
                               " permutions, and len(Atoms) == {} "
                               "!= len(ind) == {} or ind has duplicates"
                               .format(len(atoms), len(ind)))

        ind_reversed = np.zeros((len(ind)), dtype=int)
        ind_reversed[ind] = range(len(ind))
        new_symm_map = []
        for sm in self.symm_map:
            new_sm = np.array([-1] * len(atoms))
            for at_i in range(len(ind)):
                new_sm[ind_reversed[at_i]] = ind_reversed[sm[at_i]]
            new_symm_map.append(new_sm)

        self.symm_map = new_symm_map

    def todict(self):
        return {
            'name': 'FixSymmetry',
            'kwargs': {
                'atoms': self.atoms,
                'symprec': self.symprec,
                'adjust_positions': self.do_adjust_positions,
                'adjust_cell': self.do_adjust_cell,
                'verbose': self.verbose,
            },
        }
