import numpy as np

from ase.optimize.bfgs import BFGS


class RFO(BFGS):
    """RFO (Rational Function Optimizer) combined with BFGS-based Hessian
    updates.

    RFO will take quasi-Newton-like steps in quadratic regime and rational
    function damped steps outside of it. The ``damping`` factor determines
    the transition threshold between the regimes.

    Read about this algorithm here:

      | A. Banerjee, N. Adams, J. Simons, R. Shepard,
      | :doi:`Search for Stationary Points on Surfaces <10.1021/j100247a015>`
      | J. Phys. Chem. 1985, 89, 52-57.

    Note that ``damping`` is the reciprocal of coordinate scale :math:`a` from
    the reference.
    """

    # default parameters
    defaults = {**BFGS.defaults, 'damping': 1.0}

    def __init__(
        self,
        *args,
        damping: float | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *args
            Positional arguments passed to :class:`~ase.optimize.BFGS`.

        damping: float
            Determines transition threshold between quasi-Newton-like and
            rational function damped steps. The larger the value, the larger
            and stronger the damped regime. (default is 1.0 Å^-1).

        **kwargs
            Keyword arguments passed to :class:`~ase.optimize.BFGS`.
        """
        if damping is None:
            self.damping = self.defaults['damping']
        else:
            self.damping = damping

        super().__init__(*args, **kwargs)

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
        V = np.linalg.eigh(self.aug_H, subset_by_index=(0, 0))[1]
        dpos = V[:, 0][:-1] / V[:, 0][-1] / self.damping
        steplengths = self.optimizable.gradient_norm(dpos)
        self.pos0 = pos
        self.forces0 = -gradient.copy()
        return dpos, steplengths
