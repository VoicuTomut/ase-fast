# fmt: off
from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    PropertyNotPresent,
    all_properties,
)
from ase.outputs import Properties


class SinglePointCalculator(Calculator):
    """Special calculator for a single configuration.

    Used to remember the energy, force and stress for a given
    configuration.  If the positions, atomic numbers, unit cell, or
    boundary conditions are changed, then asking for
    energy/forces/stress will raise an exception."""

    name = 'unknown'

    def __init__(self, atoms, **results: Any) -> None:
        """Save energy, forces, stress, ... for the current configuration."""
        Calculator.__init__(self)
        self.results = {}
        for property, value in results.items():
            assert property in all_properties, property
            if value is None:
                continue
            if property in ['energy', 'magmom', 'free_energy']:
                self.results[property] = value
            else:
                self.results[property] = np.array(value, float)
        self.atoms = atoms.copy()

    def __str__(self) -> str:
        tokens = []
        for key, val in sorted(self.results.items()):
            if np.isscalar(val):
                txt = f'{key}={val}'
            else:
                txt = f'{key}=...'
            tokens.append(txt)
        return '{}({})'.format(self.__class__.__name__, ', '.join(tokens))

    def get_property(self, name: str, atoms=None, allow_calculation: bool = True) -> Any:
        if atoms is None:
            atoms = self.atoms
        if name not in self.results or self.check_state(atoms):
            if allow_calculation:
                raise PropertyNotImplementedError(
                    f'The property "{name}" is not available.')
            return None

        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        return result


class SinglePointKPoint:
    def __init__(self, weight: float, s: int, k: int, eps_n: np.ndarray | None = None, f_n: np.ndarray | None = None) -> None:
        self.weight = weight
        self.s = s  # spin index
        self.k = k  # k-point index
        if eps_n is None:
            eps_n = []
        self.eps_n = eps_n
        if f_n is None:
            f_n = []
        self.f_n = f_n


def arrays_to_kpoints(eigenvalues: np.ndarray, occupations: np.ndarray, weights: np.ndarray) -> list[SinglePointKPoint]:
    """Helper function for building SinglePointKPoints.

    Convert eigenvalue, occupation, and weight arrays to list of
    SinglePointKPoint objects."""
    nspins, nkpts, _nbands = eigenvalues.shape
    assert eigenvalues.shape == occupations.shape
    assert len(weights) == nkpts
    kpts = []
    for s in range(nspins):
        for k in range(nkpts):
            kpt = SinglePointKPoint(
                weight=weights[k], s=s, k=k,
                eps_n=eigenvalues[s, k], f_n=occupations[s, k])
            kpts.append(kpt)
    return kpts


class SinglePointDFTCalculator(SinglePointCalculator):
    def __init__(self, atoms,
                 efermi: float | None = None, bzkpts: np.ndarray | None = None, ibzkpts: np.ndarray | None = None, bz2ibz: np.ndarray | None = None,
                 kpts: list[SinglePointKPoint] | None = None,
                 **results: Any) -> None:
        self.bz_kpts = bzkpts
        self.ibz_kpts = ibzkpts
        self.bz2ibz = bz2ibz
        self.eFermi = efermi

        SinglePointCalculator.__init__(self, atoms, **results)
        self.kpts = kpts

    def get_fermi_level(self) -> float | None:
        """Return the Fermi-level(s)."""
        return self.eFermi

    def get_bz_to_ibz_map(self) -> np.ndarray | None:
        return self.bz2ibz

    def get_bz_k_points(self) -> np.ndarray | None:
        """Return the k-points."""
        return self.bz_kpts

    def get_number_of_spins(self) -> int | None:
        """Return the number of spins in the calculation.

        Spin-paired calculations: 1, spin-polarized calculation: 2."""
        if self.kpts is not None:
            nspin = set()
            for kpt in self.kpts:
                nspin.add(kpt.s)
            return len(nspin)
        return None

    def get_number_of_bands(self) -> int | None:
        values = {len(kpt.eps_n) for kpt in self.kpts}
        if not values:
            return None
        elif len(values) == 1:
            return values.pop()
        else:
            raise RuntimeError('Multiple array sizes')

    def get_spin_polarized(self) -> bool | None:
        """Is it a spin-polarized calculation?"""
        nos = self.get_number_of_spins()
        if nos is not None:
            return nos == 2
        return None

    def get_ibz_k_points(self) -> np.ndarray | None:
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibz_kpts

    def get_kpt(self, kpt: int = 0, spin: int = 0) -> SinglePointKPoint | None:
        if self.kpts is not None:
            counter = 0
            for kpoint in self.kpts:
                if kpoint.s == spin:
                    if kpt == counter:
                        return kpoint
                    counter += 1
        return None

    def get_k_point_weights(self) -> np.ndarray | None:
        """ Retunrs the weights of the k points """
        if self.kpts is not None:
            weights = []
            for kpoint in self.kpts:
                if kpoint.s == 0:
                    weights.append(kpoint.weight)
            return np.array(weights)
        return None

    def get_occupation_numbers(self, kpt: int = 0, spin: int = 0) -> np.ndarray | None:
        """Return occupation number array."""
        kpoint = self.get_kpt(kpt, spin)
        if kpoint is not None:
            if len(kpoint.f_n):
                return kpoint.f_n
        return None

    def get_eigenvalues(self, kpt: int = 0, spin: int = 0) -> np.ndarray | None:
        """Return eigenvalue array."""
        kpoint = self.get_kpt(kpt, spin)
        if kpoint is not None:
            return kpoint.eps_n
        return None

    def get_homo_lumo(self) -> tuple[float, float]:
        """Return HOMO and LUMO energies."""
        if self.kpts is None:
            raise RuntimeError('No kpts')
        eH = -np.inf
        eL = np.inf
        for spin in range(self.get_number_of_spins()):
            homo, lumo = self.get_homo_lumo_by_spin(spin)
            eH = max(eH, homo)
            eL = min(eL, lumo)
        return eH, eL

    def get_homo_lumo_by_spin(self, spin: int = 0) -> tuple[float, float]:
        """Return HOMO and LUMO energies for a given spin."""
        if self.kpts is None:
            raise RuntimeError('No kpts')
        for kpt in self.kpts:
            if kpt.s == spin:
                break
        else:
            raise RuntimeError(f'No k-point with spin {spin}')
        if self.eFermi is None:
            raise RuntimeError('Fermi level is not available')
        eH = -1.e32
        eL = 1.e32
        for kpt in self.kpts:
            if kpt.s == spin:
                for e in kpt.eps_n:
                    if e <= self.eFermi:
                        eH = max(eH, e)
                    else:
                        eL = min(eL, e)
        return eH, eL

    def properties(self) -> Properties:
        return OutputPropertyWrapper(self).properties()


def propertygetter(func: Any) -> Any:
    from functools import wraps

    @wraps(func)
    def getter(self):
        value = func(self)
        if value is None:
            raise PropertyNotPresent(func.__name__)
        return value
    return cached_property(getter)


class OutputPropertyWrapper:
    def __init__(self, calc: SinglePointDFTCalculator) -> None:
        self.calc = calc

    @propertygetter
    def nspins(self) -> int:
        return self.calc.get_number_of_spins()

    @propertygetter
    def nbands(self) -> int:
        return self.calc.get_number_of_bands()

    @propertygetter
    def nkpts(self) -> int:
        return len(self.calc.kpts) // self.nspins

    def _build_eig_occ_array(self, getter) -> np.ndarray | None:
        arr = np.empty((self.nspins, self.nkpts, self.nbands))
        for s in range(self.nspins):
            for k in range(self.nkpts):
                value = getter(spin=s, kpt=k)
                if value is None:
                    return None
                arr[s, k, :] = value
        return arr

    @propertygetter
    def eigenvalues(self) -> np.ndarray:
        return self._build_eig_occ_array(self.calc.get_eigenvalues)

    @propertygetter
    def occupations(self) -> np.ndarray:
        return self._build_eig_occ_array(self.calc.get_occupation_numbers)

    @propertygetter
    def fermi_level(self) -> float:
        return self.calc.get_fermi_level()

    @propertygetter
    def kpoint_weights(self) -> np.ndarray:
        return self.calc.get_k_point_weights()

    @propertygetter
    def ibz_kpoints(self) -> np.ndarray:
        return self.calc.get_ibz_k_points()

    def properties(self) -> Properties:
        dct = {}
        for name in ['eigenvalues', 'occupations', 'fermi_level',
                     'kpoint_weights', 'ibz_kpoints']:
            try:
                value = getattr(self, name)
            except PropertyNotPresent:
                pass
            else:
                dct[name] = value

        for name, value in self.calc.results.items():
            dct[name] = value

        return Properties(dct)
