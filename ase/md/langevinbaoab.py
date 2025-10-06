"""Time integrator using Langevin for positions and Langevin-Hoover
(variable volume, fixed cell shape only) for cell with BAOAB time propagation

BAOAB algorithm from Leimkuhler and Matthews "Robust and efficient
configurational molecular sampling via Langevin dynamics",
J. Chem. Phys. 138 174102 (2013).
https://doi.org/10.1063/1.4802990

There is some evidence that other time integration schemes, e.g. BAOA,
maybe be better (https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00585),
and it may be straightforward to add these, but none are not currently
supported .

Langevin-Hoover from Quigley and Probert "Langevin dynamics in constant
pressure extended systems", J. Chem. Phys 120 11432 (2004).
https://doi.org/10.1063/1.1755657

Parameters
----------
atoms: Atoms
    atoms object for dynaics
timestep: float
    timestep (ASE native units) for time propagation
temperature_K: float, optional
    constant temperature in K to apply.  Enables N[VP]T.
externalstress: float, ndarray(3), narray(6), narray((3, 3)) optional
    constant stress (i.e. negative of pressure, so _negative_ values lead to
    compression) to apply, in ASE native units. Enables NP[TH].  Note that
    barostat will keep mean stress _including kinetic (i.e. ideal gas)
    contribution_ equal to this value.  Only scalars are allowed if
    hydrostatic is True
T_tau: float, optional
    time constant for temperature Langevin. Defaults to 50 * `timestep` if not
    specified
P_tau: float, optional
    time constant for pressure (variable cell) Langevin. Defaults to
    20 * `T_tau` if T_tau is provided, otherwise 1000 * `timestep`.  A value
    of 0 turns off Langevin thermalization of cell DOF
P_mass: float, optional
    mass used for cell volume dynamics. Default heuristic value to aim for
    fluctuation period of P_tau / 4.  Default heuristic will fail if `P_tau`
    is not > 0.
P_mass_factor: float, default 1.0
    factor to multiply heuristic P_mass
hydrostatic: bool, default False
    allow only hydrostaic strain.
initial_nsteps: int, default 0
    initial nsteps to set (for sensible output in continuations)
**kwargs: dict
    additional ase.md.md.MolecularDynamics kwargs
"""

import warnings

import numpy as np
from scipy.linalg import expm

from ase import units
from ase.md.md import MolecularDynamics
from ase.stress import voigt_6_to_full_3x3_stress


class LangevinBAOAB(MolecularDynamics):
    def __init__(
        self,
        atoms,
        timestep,
        *,
        temperature_K=None,
        externalstress=None,
        T_tau=None,
        P_tau=None,
        P_mass=None,
        P_mass_factor=None,
        rng=None,
        hydrostatic=False,
        initial_nsteps=0,
        **kwargs,
    ):
        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)

        self.externalstress = externalstress
        self.P_mass = P_mass
        self.P_mass_factor = P_mass_factor if P_mass_factor is not None else 1.0
        self.rng = rng

        self.nsteps = initial_nsteps

        if temperature_K is not None:
            # run constant T, need rng and T_tau
            if self.rng is None:
                raise RuntimeError(
                    f"Fixed temperature requires `rng`, got '{rng}'"
                )
            if T_tau is None:
                raise RuntimeError(
                    f"Fixed temperature requires `T_tau`, got '{T_tau}'"
                )
        self.T_tau = T_tau

        self.hydrostatic = None
        if self.externalstress is not None:
            self.hydrostatic = hydrostatic

            # promote to ndarray to simplify code below
            self.externalstress = np.asarray(self.externalstress).reshape((-1))
            s = self.externalstress.shape

            # run NPH or NPT, need P_mass, and optionally rng and temperature_K
            if hydrostatic:
                # external stress must be scalar
                if s != (1,):
                    raise ValueError(
                        'externalstress must be scalar when hydrostatic, '
                        f"got '{self.externalstress}' with shape "
                        f'{self.externalstress.shape}'
                    )
                self.externalstress = self.externalstress[0]
            else:
                # external stress must end up as 3x3 matrix
                if s == (1,):
                    self.externalstress = self.externalstress * np.identity(3)
                elif s == (3,):
                    self.externalstress = np.diag(self.externalstress)
                elif s == (6,):
                    self.externalstress = voigt_6_to_full_3x3_stress(
                        self.externalstress
                    )
                elif s != (3, 3):
                    raise ValueError(
                        'externalstress must be scalar, 3-vector (diagonal), '
                        '6-vector (Voigt), or 3x3 matrix, '
                        f"'{self.externalstress}' with shape "
                        f'{self.externalstress.shape}'
                    )

            if P_tau is None:
                if self.T_tau is not None:
                    P_tau = 20.0 * self.T_tau
                    warnings.warn(
                        'Got externalstress but missing P_tau, got '
                        f'T_tau, defaulting to 20 * T_tau = {P_tau}'
                    )
                else:
                    P_tau = 1000.0 * self.dt
                    warnings.warn(
                        'Got externalstress but missing P_tau and '
                        f'T_tau, defaulting to 1000 * timestep = {P_tau}'
                    )
        self.P_tau = P_tau

        # default contribution to effective gamma used in _BAOAB_OU that comes
        # from barostat from 2nd term in RHS of Quigley Eq. (5b)
        self.gamma_mod = 0.0

        if self.externalstress is not None:
            self.p_eps = 0.0
            # Hope that ASE get_number_of_degrees_of_freedom gives correct value.
            # It's not, for example, completely obvious what should be
            # done about the 3 overall translation DOFs, since conventional
            # Langevin does not actually preserve those (i.e. violates
            # conservation of momentum). See, e.g.,
            #     https://doi.org/10.1063/5.0286750
            # for discussion of variants, e.g. DPD pairwise-force thermostat
            if len(self.atoms.constraints()) != 0:
                warnings.warn("WARNING: LangevinBAOAB has not been "
                              "tested with constraints")
            self.Ndof = self.atoms.get_number_of_degrees_of_freedom()

        self.set_temperature(temperature_K, from_init=True)

        # initialize forces and stresses
        self._update_accel()
        if self.externalstress is not None:
            self._update_force_eps()

    def set_temperature(self, temperature_K, from_init=False):
        """Set the internal parameters that depend on temperature

        Parameters
        ----------
        temperature_K: float
            temperature in K
        """
        self.temperature_K = temperature_K

        # set P_mass with heuristic
        #
        # originally tried expression from Quigley Eq. 17
        #    W = 3 N k_B T / (2 pi / tau)^2
        # didn't work at all
        #
        # instead, using empirical value based on tests
        # of various P_mass, supercell sizes
        #
        # supercell of 4 atom Al FCC cell
        # VARY P_mass: looks like tau \prop sqrt(P_mass)
        # sc 3 P_mass 1000  T 300 period 34.01360544217687
        # sc 3 P_mass 10000 T 300 period 91.74311926605505
        # VARY sc: looks like tau \prop 1/sqrt(N^1/3)
        # sc 2 P_mass 10000.0 T 400 period 104.16666666666667 (32 atoms)
        # sc 3 P_mass 10000.0 T 400 period 92.5925925925926  (108 atoms)
        # sc 4 P_mass 10000.0 T 400 period 66.66666666666667 (256 atoms)
        # tau = C * sqrt(P_mass) / N**(1/6)
        # 66 fs for N = 4 * 4^3 = 256, P_mass = 10^4
        # 66 fs = C * 10000**0.5 / 256.0**(1.0/6.0)
        # C = 66 fs / (10000**0.5 / 256.0**(1.0/6.0))
        # C = 1.6630957858612323 fs
        # P_mass = ((tau / C) * N**(1/6)) ** 2

        if self.externalstress is not None and self.P_mass is None:
            if not self.P_tau > 0:
                raise ValueError('Heuristic used for P_mass requires P_tau > 0')
            C = 1.66 * units.fs
            self.barostat_mass_use = (
                ((self.P_tau / 4.0) / C) * (len(self.atoms) ** (1.0 / 6.0))
            ) ** 2
            self.barostat_mass_use *= self.P_mass_factor
            if from_init:
                warnings.warn(
                    'Using heuristic P_mass '
                    f'{self.barostat_mass_use} '
                    f'from P_tau {self.P_tau}'
                )
        else:
            self.barostat_mass_use = self.P_mass

        # store quantities for BAOAB
        self.gamma = 0.0
        if self.temperature_K is not None:
            self.BAOAB_prefactor = 0.0
            if self.T_tau != 0:
                self.gamma = 1.0 / self.T_tau
                # sigma from before Eq. 4 of Leimkuhler
                sigma = np.sqrt(
                    2.0 * self.gamma * units.kB * self.temperature_K
                )
                # prefactor from after Eq. 6
                self.BAOAB_prefactor = (
                    sigma / np.sqrt(2.0 * self.gamma)
                ) * np.sqrt(1.0 - np.exp(-2.0 * self.gamma * self.dt))
                # does not include sqrt(mass), since that is different for
                # each atom type

        # initialize deformation gradient dynamical variables
        if self.externalstress is not None:
            self.barostat_gamma = 0.0
            if self.P_tau != 0:
                if self.temperature_K is None:
                    raise RuntimeError(
                        'Got thermalized barostat with P_tau '
                        f'{self.P_tau} != 0, also need '
                        'temperature_K which was not specified'
                    )
                self.barostat_gamma = 1.0 / self.P_tau
                sigma = np.sqrt(
                    2.0 * self.barostat_gamma * units.kB * self.temperature_K
                )
                self._barostat_BAOAB_prefactor = (
                    (sigma / np.sqrt(2.0 * self.barostat_gamma))
                    * np.sqrt(
                        1.0 - np.exp(-2.0 * self.barostat_gamma * self.dt)
                    )
                    * np.sqrt(self.barostat_mass_use)
                )
                # _does_ include sqrt(mass) factor

    def _update_accel(self):
        """Update position-acceleration from current positions via forces"""
        self.accel = (self.atoms.get_forces().T / self.atoms.get_masses()).T

    def _update_force_eps(self):
        """Update cell force from current positions via stress"""
        volume = self.atoms.get_volume()
        if self.hydrostatic:
            KE = self.atoms.get_kinetic_energy()
            Tr_virial = -volume * np.trace(self.atoms.get_stress(voigt=False))
            X = 1.0 / (3.0 * volume) * (2.0 * KE + Tr_virial)

            # NB explicit dphi/dV term in Quigley Eq. 6 is old fashioned
            # explicit volume dependence for long range tails.  Stress/pressure
            # comes in via sum r . f, which is
            # Tr[virial] = - volume * Tr[stress]
            self.force_eps = (
                3.0 * volume * (X + self.externalstress)
                + (3.0 / self.Ndof) * 2.0 * KE
            )
        else:
            mom = self.atoms.get_momenta()
            kinetic_stress_contrib = (mom.T / self.atoms.get_masses()) @ mom
            virial = -volume * self.atoms.get_stress(voigt=False)
            X = (1.0 / volume) * (kinetic_stress_contrib + virial)

            self.force_eps = volume * (X + self.externalstress) + (
                1.0 / self.Ndof
            ) * np.trace(kinetic_stress_contrib) * np.identity(3)

    def _BAOAB_B(self):
        """Do a BAOAB B (velocity) half step"""
        dvel = 0.5 * self.dt * self.accel
        self.atoms.set_velocities(self.atoms.get_velocities() + dvel)

    def _barostat_BAOAB_B(self):
        """Do a barostat BAOAB B (cell momentum) half step"""
        self.p_eps += 0.5 * self.dt * self.force_eps

    def _BAOAB_A(self):
        """Do a BAOAB A (position) half step"""
        self.atoms.positions += 0.5 * self.dt * self.atoms.get_velocities()

    def _barostat_BAOAB_A(self):
        """Do a barostat BAOAB A (cell volume) half step"""
        if self.hydrostatic:
            volume = self.atoms.get_volume()
            new_volume = volume * np.exp(
                0.5 * self.dt * 3.0 * self.p_eps / self.barostat_mass_use
            )
            new_cell = self.atoms.cell * (new_volume / volume) ** (1.0 / 3.0)
        else:
            new_cell = (
                self.atoms.cell
                @ expm(0.5 * self.dt * self.p_eps / self.barostat_mass_use).T
            )

        self.atoms.set_cell(new_cell, True)

    def _BAOAB_OU(self, drag_gamma_mod=0.0):
        """Do a BAOAB Ornstein-Uhlenbeck position Langevin full step

        Parameters
        ----------
        drag_gamma_mod: float, default 0
            additional contribution to effective gamma used for drag on
            velocities, e.g. from Langevin-Hoover
        """
        if self.gamma == 0 and drag_gamma_mod == 0:
            return

        vel = self.atoms.get_velocities()

        if self.hydrostatic:
            vel *= np.exp(-(self.gamma + drag_gamma_mod) * self.dt)
        else:
            vel = (
                vel
                @ expm(
                    -(self.gamma * np.identity(3) + drag_gamma_mod) * self.dt
                ).T
            )

        if self.gamma != 0:
            # here we divide by sqrt(m), since Leimkuhler definition includes
            # sqrt(m) in numerator but that's for momentum, so for velocity we
            # divide by m, i.e. net 1/sqrt(m)
            vel_shape = self.atoms.positions.shape
            masses = self.atoms.get_masses()
            vel += (
                self.BAOAB_prefactor
                * (self.rng.normal(size=vel_shape).T / np.sqrt(masses)).T
            )

        self.atoms.set_velocities(vel)

    def _barostat_BAOAB_OU(self):
        """Do a barostat BAOAB Ornstein-Uhlenbeck cell volume Langevin full
        step"""
        if self.barostat_gamma == 0:
            return

        self.p_eps *= np.exp(-self.barostat_gamma * self.dt)

        if self.hydrostatic:
            self.p_eps += self._barostat_BAOAB_prefactor * self.rng.normal()
        else:
            random_force = self.rng.normal(size=(3, 3))
            # symmetrize to avoid rotation
            random_force += random_force.T
            random_force /= 2
            self.p_eps += self._barostat_BAOAB_prefactor * random_force

    def _barostat_BAOAB_OU_gamma_mod(self):
        """Compute contribution to drag effective gamma applied to atom
        velocities due to barostat velocity
        """
        if self.hydrostatic:
            return (1.0 + 3.0 / self.Ndof) * self.p_eps / self.barostat_mass_use
        else:
            return (
                self.p_eps + (1.0 / self.Ndof) * np.trace(self.p_eps)
            ) / self.barostat_mass_use

    def step(self):
        """Do a time step"""
        if self.externalstress is not None:
            self._barostat_BAOAB_B()
        self._BAOAB_B()  # half step vel

        if self.externalstress is not None:
            self._barostat_BAOAB_A()
        self._BAOAB_A()  # half step pos

        if self.externalstress is not None:
            self.gamma_mod = self._barostat_BAOAB_OU_gamma_mod()
        self._BAOAB_OU(self.gamma_mod)  # OU vel
        if self.externalstress is not None:
            self._barostat_BAOAB_OU()

        self._BAOAB_A()  # half step pos
        if self.externalstress is not None:
            self._barostat_BAOAB_A()

        self._update_accel()  # update accel from final pos
        if self.externalstress is not None:
            self._update_force_eps()  # update cell force

        self._BAOAB_B()  # half step vel
        if self.externalstress is not None:
            self._barostat_BAOAB_B()

        # self.nsteps += 1
