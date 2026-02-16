# fmt: off

from pathlib import Path
from typing import IO

import numpy as np

from ase import Atoms
from ase.optimize.bfgs import BFGS


class RFO(BFGS):
    """RFO (Rational Function Optimizer) combined with BFGS-based Hessian
    updates.

    RFO will take quasi-Newton-like steps in quadratic regime and rational
    function damped steps outside of it. The ``damping`` factor determines
    the transition threshold between the regimes.
    """
    # default parameters
    defaults = {**BFGS.defaults, 'damping': 1.0}

    def __init__(
        self,
        atoms: Atoms,
        restart: str | Path | None = None,
        logfile: IO | str | Path | None = '-',
        trajectory: str | Path | None = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        damping: float | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str | Path | None
            JSON file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: str or Path
            Trajectory file used to store optimisation path.

        logfile: file object, Path, or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        damping: float
            Determines transition threshold between quasi-Newton-like and
            rational function damped steps. The larger the value, the larger
            and stronger the damped regime. (default is 1.0 Å^-1).

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.
        """
        if damping is None:
            self.damping = self.defaults['damping']
        else:
            self.damping = damping

        super().__init__(
            atoms=atoms, restart=restart,
            logfile=logfile, trajectory=trajectory,
            append_trajectory=append_trajectory,
            maxstep=maxstep, alpha=alpha,
            **kwargs)

    def initialize(self):
        # Initialize Hessian
        super().initialize()
        n = len(self.H0)
        self.aug_H = np.zeros((n + 1, n + 1))

    def read(self):
        # Read Hessian
        super().read()
        n = len(self.H)
        self.aug_H = np.zeros((n + 1, n + 1))

    def prepare_step(self, pos, gradient):
        """Compute step from first eigenvector of gradient-augmented Hessian"""
        # Update Hessian BFGS-style
        self.update(pos, -gradient, self.pos0, self.forces0)
        self.aug_H[:-1, :-1] = self.H / self.damping**2
        self.aug_H[-1, :-1] = gradient / self.damping
        self.aug_H[:-1, -1] = gradient / self.damping
        V = np.linalg.eigh(self.aug_H)[1]
        dpos = V[:, 0][:-1] / V[:, 0][-1] / self.damping
        steplengths = self.optimizable.gradient_norm(dpos)
        self.pos0 = pos
        self.forces0 = -gradient.copy()
        return dpos, steplengths
