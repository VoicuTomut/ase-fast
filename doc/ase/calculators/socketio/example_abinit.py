from ase.build import bulk
from ase.calculators.abinit import Abinit
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

# Modify this line to suit your needs:
pseudopotentials = {'Si': '14-Si.LDA.fhi'}

atoms = bulk('Si')
atoms.rattle(stdev=0.1, seed=42)

# Implementation note: Socket-driven calculations in Abinit inherit several
# controls for from the ordinary cell optimization code.  We have to hack those
# variables in order for Abinit not to decide that the calculation converged:
boilerplate_kwargs = dict(
    tolmxf=1e-300,  # Prevent Abinit from thinking we "converged"
    ntime=100_000,  # Allow "infinitely" many iterations in Abinit
    ecutsm=0.5,  # Smoothing PW cutoff energy (mandatory for cell optimization)
)

kwargs = dict(
    ecut=5 * 27.3,
    tolvrs=1e-8,
    kpts=[2, 2, 2],
    **boilerplate_kwargs,
)

abinit = Abinit(directory='abinit-calc', **kwargs)
opt = BFGS(FrechetCellFilter(atoms), trajectory='opt.traj')

with abinit.socketio(unixsocket='ase-abinit') as atoms.calc:
    opt.run(fmax=0.01)
