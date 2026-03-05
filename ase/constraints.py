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


class Hookean(FixConstraint):
    """Applies a Hookean restorative force between a pair of atoms, an atom
    and a point, or an atom and a plane."""

    def __init__(self, a1, a2, k, rt=None):
        """Forces two atoms to stay close together by applying no force if
        they are below a threshold length, rt, and applying a Hookean
        restorative force when the distance between them exceeds rt. Can
        also be used to tether an atom to a fixed point in space or to a
        distance above a plane.

        a1 : int
           Index of atom 1
        a2 : one of three options
           1) index of atom 2
           2) a fixed point in cartesian space to which to tether a1
           3) a plane given as (A, B, C, D) in A x + B y + C z + D = 0.
        k : float
           Hooke's law (spring) constant to apply when distance
           exceeds threshold_length. Units of eV A^-2.
        rt : float
           The threshold length below which there is no force. The
           length is 1) between two atoms, 2) between atom and point.
           This argument is not supplied in case 3. Units of A.

        If a plane is specified, the Hooke's law force is applied if the atom
        is on the normal side of the plane. For instance, the plane with
        (A, B, C, D) = (0, 0, 1, -7) defines a plane in the xy plane with a z
        intercept of +7 and a normal vector pointing in the +z direction.
        If the atom has z > 7, then a downward force would be applied of
        k * (atom.z - 7). The same plane with the normal vector pointing in
        the -z direction would be given by (A, B, C, D) = (0, 0, -1, 7).

        References:

           Andrew A. Peterson,  Topics in Catalysis volume 57, pages40–53 (2014)
           https://link.springer.com/article/10.1007%2Fs11244-013-0161-8
        """

        if isinstance(a2, int):
            self._type = 'two atoms'
            self.indices = [a1, a2]
        elif len(a2) == 3:
            self._type = 'point'
            self.index = a1
            self.origin = np.array(a2)
        elif len(a2) == 4:
            self._type = 'plane'
            self.index = a1
            self.plane = a2
        else:
            raise RuntimeError('Unknown type for a2')
        self.threshold = rt
        self.spring = k

    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        dct = {'name': 'Hookean'}
        dct['kwargs'] = {'rt': self.threshold,
                         'k': self.spring}
        if self._type == 'two atoms':
            dct['kwargs']['a1'] = self.indices[0]
            dct['kwargs']['a2'] = self.indices[1]
        elif self._type == 'point':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.origin
        elif self._type == 'plane':
            dct['kwargs']['a1'] = self.index
            dct['kwargs']['a2'] = self.plane
        else:
            raise NotImplementedError(f'Bad type: {self._type}')
        return dct

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d < 0:
                return
            magnitude = self.spring * d
            direction = - np.array((A, B, C)) / np.linalg.norm((A, B, C))
            forces[self.index] += direction * magnitude
            return
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            magnitude = self.spring * (bondlength - self.threshold)
            direction = displace / np.linalg.norm(displace)
            if self._type == 'two atoms':
                forces[self.indices[0]] += direction * magnitude
                forces[self.indices[1]] -= direction * magnitude
            else:
                forces[self.index] += direction * magnitude

    def adjust_potential_energy(self, atoms):
        """Returns the difference to the potential energy due to an active
        constraint. (That is, the quantity returned is to be added to the
        potential energy.)"""
        positions = atoms.positions
        if self._type == 'plane':
            A, B, C, D = self.plane
            x, y, z = positions[self.index]
            d = ((A * x + B * y + C * z + D) /
                 np.sqrt(A**2 + B**2 + C**2))
            if d > 0:
                return 0.5 * self.spring * d**2
            else:
                return 0.
        if self._type == 'two atoms':
            p1, p2 = positions[self.indices]
        elif self._type == 'point':
            p1 = positions[self.index]
            p2 = self.origin
        displace, _ = find_mic(p2 - p1, atoms.cell, atoms.pbc)
        bondlength = np.linalg.norm(displace)
        if bondlength > self.threshold:
            return 0.5 * self.spring * (bondlength - self.threshold)**2
        else:
            return 0.

    def get_indices(self):
        if self._type == 'two atoms':
            return self.indices
        elif self._type == 'point':
            return self.index
        elif self._type == 'plane':
            return self.index

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        if self._type == 'two atoms':
            newa = [-1, -1]  # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                for i, a in enumerate(self.indices):
                    if old == a:
                        newa[i] = new
            if newa[0] == -1 or newa[1] == -1:
                raise IndexError('Constraint not part of slice')
            self.indices = newa
        elif (self._type == 'point') or (self._type == 'plane'):
            newa = -1   # Signal error
            for new, old in slice2enlist(ind, len(atoms)):
                if old == self.index:
                    newa = new
                    break
            if newa == -1:
                raise IndexError('Constraint not part of slice')
            self.index = newa

    def __repr__(self):
        if self._type == 'two atoms':
            return 'Hookean(%d, %d)' % tuple(self.indices)
        elif self._type == 'point':
            return 'Hookean(%d) to cartesian' % self.index
        else:
            return 'Hookean(%d) to plane' % self.index


class ExternalForce(FixConstraint):
    """Constraint object for pulling two atoms apart by an external force.

    You can combine this constraint for example with FixBondLength but make
    sure that *ExternalForce* comes first in the list if there are overlaps
    between atom1-2 and atom3-4:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4 = atoms[:4]
    >>> fext = 1.0
    >>> con1 = ExternalForce(atom1, atom2, f_ext)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    see ase/test/external_force.py"""

    def __init__(self, a1, a2, f_ext):
        self.indices = [a1, a2]
        self.external_force = f_ext

    def get_removed_dof(self, atoms):
        return 0

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        force = self.external_force * dist / np.linalg.norm(dist)
        forces[self.indices] += (force, -force)

    def adjust_potential_energy(self, atoms):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        return -np.linalg.norm(dist) * self.external_force

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'ExternalForce(%d, %d, %f)' % (self.indices[0],
                                              self.indices[1],
                                              self.external_force)

    def todict(self):
        return {'name': 'ExternalForce',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'f_ext': self.external_force}}


class MirrorForce(FixConstraint):
    """Constraint object for mirroring the force between two atoms.

    This class is designed to find a transition state with the help of a
    single optimization. It can be used if the transition state belongs to a
    bond breaking reaction. First the given bond length will be fixed until
    all other degrees of freedom are optimized, then the forces of the two
    atoms will be mirrored to find the transition state. The mirror plane is
    perpendicular to the connecting line of the atoms. Transition states in
    dependence of the force can be obtained by stretching the molecule and
    fixing its total length with *FixBondLength* or by using *ExternalForce*
    during the optimization with *MirrorForce*.

    Parameters
    ----------
    a1: int
        First atom index.
    a2: int
        Second atom index.
    max_dist: float
        Upper limit of the bond length interval where the transition state
        can be found.
    min_dist: float
        Lower limit of the bond length interval where the transition state
        can be found.
    fmax: float
        Maximum force used for the optimization.

    Notes
    -----
    You can combine this constraint for example with FixBondLength but make
    sure that *MirrorForce* comes first in the list if there are overlaps
    between atom1-2 and atom3-4:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4 = atoms[:4]
    >>> con1 = MirrorForce(atom1, atom2)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    """

    def __init__(self, a1, a2, max_dist=2.5, min_dist=1., fmax=0.1):
        self.indices = [a1, a2]
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.fmax = fmax

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        d = np.linalg.norm(dist)
        if (d < self.min_dist) or (d > self.max_dist):
            # Stop structure optimization
            forces[:] *= 0
            return
        dist /= d
        df = np.subtract.reduce(forces[self.indices])
        f = df.dot(dist)
        con_saved = atoms.constraints
        try:
            con = [con for con in con_saved
                   if not isinstance(con, MirrorForce)]
            atoms.set_constraint(con)
            forces_copy = atoms.get_forces()
        finally:
            atoms.set_constraint(con_saved)
        df1 = -1 / 2. * f * dist
        forces_copy[self.indices] += (df1, -df1)
        # Check if forces would be converged if the bond with mirrored forces
        # would also be fixed
        if (forces_copy**2).sum(axis=1).max() < self.fmax**2:
            factor = 1.
        else:
            factor = 0.
        df1 = -(1 + factor) / 2. * f * dist
        forces[self.indices] += (df1, -df1)

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint

        """
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'MirrorForce(%d, %d, %f, %f, %f)' % (
            self.indices[0], self.indices[1], self.max_dist, self.min_dist,
            self.fmax)

    def todict(self):
        return {'name': 'MirrorForce',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'max_dist': self.max_dist,
                           'min_dist': self.min_dist, 'fmax': self.fmax}}


class MirrorTorque(FixConstraint):
    """Constraint object for mirroring the torque acting on a dihedral
    angle defined by four atoms.

    This class is designed to find a transition state with the help of a
    single optimization. It can be used if the transition state belongs to a
    cis-trans-isomerization with a change of dihedral angle. First the given
    dihedral angle will be fixed until all other degrees of freedom are
    optimized, then the torque acting on the dihedral angle will be mirrored
    to find the transition state. Transition states in
    dependence of the force can be obtained by stretching the molecule and
    fixing its total length with *FixBondLength* or by using *ExternalForce*
    during the optimization with *MirrorTorque*.

    This constraint can be used to find
    transition states of cis-trans-isomerization.

    a1    a4
    |      |
    a2 __ a3

    Parameters
    ----------
    a1: int
        First atom index.
    a2: int
        Second atom index.
    a3: int
        Third atom index.
    a4: int
        Fourth atom index.
    max_angle: float
        Upper limit of the dihedral angle interval where the transition state
        can be found.
    min_angle: float
        Lower limit of the dihedral angle interval where the transition state
        can be found.
    fmax: float
        Maximum force used for the optimization.

    Notes
    -----
    You can combine this constraint for example with FixBondLength but make
    sure that *MirrorTorque* comes first in the list if there are overlaps
    between atom1-4 and atom5-6:

    >>> from ase.build import bulk

    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> atom1, atom2, atom3, atom4, atom5, atom6 = atoms[:6]
    >>> con1 = MirrorTorque(atom1, atom2, atom3, atom4)
    >>> con2 = FixBondLength(atom5, atom6)
    >>> atoms.set_constraint([con1, con2])

    """

    def __init__(self, a1, a2, a3, a4, max_angle=2 * np.pi, min_angle=0.,
                 fmax=0.1):
        self.indices = [a1, a2, a3, a4]
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.fmax = fmax

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        angle = atoms.get_dihedral(self.indices[0], self.indices[1],
                                   self.indices[2], self.indices[3])
        angle *= np.pi / 180.
        if (angle < self.min_angle) or (angle > self.max_angle):
            # Stop structure optimization
            forces[:] *= 0
            return
        p = atoms.positions[self.indices]
        f = forces[self.indices]

        f0 = (f[1] + f[2]) / 2.
        ff = f - f0
        p0 = (p[2] + p[1]) / 2.
        m0 = np.cross(p[1] - p0, ff[1]) / (p[1] - p0).dot(p[1] - p0)
        fff = ff - np.cross(m0, p - p0)
        d1 = np.cross(np.cross(p[1] - p0, p[0] - p[1]), p[1] - p0) / \
            (p[1] - p0).dot(p[1] - p0)
        d2 = np.cross(np.cross(p[2] - p0, p[3] - p[2]), p[2] - p0) / \
            (p[2] - p0).dot(p[2] - p0)
        omegap1 = (np.cross(d1, fff[0]) / d1.dot(d1)).dot(p[1] - p0) / \
            np.linalg.norm(p[1] - p0)
        omegap2 = (np.cross(d2, fff[3]) / d2.dot(d2)).dot(p[2] - p0) / \
            np.linalg.norm(p[2] - p0)
        omegap = omegap1 + omegap2
        con_saved = atoms.constraints
        try:
            con = [con for con in con_saved
                   if not isinstance(con, MirrorTorque)]
            atoms.set_constraint(con)
            forces_copy = atoms.get_forces()
        finally:
            atoms.set_constraint(con_saved)
        df1 = -1 / 2. * omegap * np.cross(p[1] - p0, d1) / \
            np.linalg.norm(p[1] - p0)
        df2 = -1 / 2. * omegap * np.cross(p[2] - p0, d2) / \
            np.linalg.norm(p[2] - p0)
        forces_copy[self.indices] += (df1, [0., 0., 0.], [0., 0., 0.], df2)
        # Check if forces would be converged if the dihedral angle with
        # mirrored torque would also be fixed
        if (forces_copy**2).sum(axis=1).max() < self.fmax**2:
            factor = 1.
        else:
            factor = 0.
        df1 = -(1 + factor) / 2. * omegap * np.cross(p[1] - p0, d1) / \
            np.linalg.norm(p[1] - p0)
        df2 = -(1 + factor) / 2. * omegap * np.cross(p[2] - p0, d2) / \
            np.linalg.norm(p[2] - p0)
        forces[self.indices] += (df1, [0., 0., 0.], [0., 0., 0.], df2)

    def index_shuffle(self, atoms, ind):
        # See docstring of superclass
        indices = []
        for new, old in slice2enlist(ind, len(atoms)):
            if old in self.indices:
                indices.append(new)
        if len(indices) == 0:
            raise IndexError('All indices in MirrorTorque not part of slice')
        self.indices = np.asarray(indices, int)

    def __repr__(self):
        return 'MirrorTorque(%d, %d, %d, %d, %f, %f, %f)' % (
            self.indices[0], self.indices[1], self.indices[2],
            self.indices[3], self.max_angle, self.min_angle, self.fmax)

    def todict(self):
        return {'name': 'MirrorTorque',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1],
                           'a3': self.indices[2], 'a4': self.indices[3],
                           'max_angle': self.max_angle,
                           'min_angle': self.min_angle, 'fmax': self.fmax}}


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
