import numpy as np
import pytest

from ase.utils.abc import Optimizable
import ase.optimize


class BoothFunctionOptimizable(Optimizable):
    """Optimizable based on the “Booth” function.

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self, x0):
        self.xy = np.array(x0)

    def get_x(self):
        return self.xy.copy()

    def set_x(self, x):
        self.xy[:] = x

    @staticmethod
    def ab(x, y):
        return x + 2 * y - 7, 2 * x + y - 5

    def get_value(self):
        a, b = self.ab(*self.xy)
        return a * a + b * b

    def get_gradient(self):
        a, b = self.ab(*self.xy)
        # XXX negative gradient
        return -np.array([2 * a + 4 * b, 4 * a + 2 * b])

    def iterimages(self):
        return iter([])

    def ndofs(self):
        return len(self.xy)

    def gradient_norm(self, gradient):
        return np.linalg.norm(gradient)


optimizers = [
    'BFGS',
    'BFGSLineSearch',
    'MDMin',
    'FIRE',
    'FIRE2',
    'LBFGS',
    # 'LBFGSLineSearch',
    # 'GoodOldQuasiNewton',
    # 'GPMin',
    # 'ODE12r',
]


@pytest.mark.parametrize('optname', optimizers)
def test_booth(optname):
    x0 = [1.234, 2.345]
    target = BoothFunctionOptimizable(x0)

    eps = 1e-8
    optcls = getattr(ase.optimize, optname)
    with optcls(target) as opt:
        opt.run(fmax=eps)

    assert target.xy == pytest.approx([1, 3], abs=eps)
    assert target.get_value() == pytest.approx(0, abs=eps**2)
