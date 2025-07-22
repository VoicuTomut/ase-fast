from ase.build import molecule
from ase.calculators.aims import Aims
from ase.optimize import BFGS

# Note that FHI-aim support for the i-PI protocol must be specifically
# enabled at compile time, e.g.: make -f Makefile.ipi ipi.mpi

# This example uses INET; see other examples for how to use UNIX sockets.
port = 31415

atoms = molecule('H2O', vacuum=3.0)
atoms.rattle(stdev=0.1)

aims = Aims(directory='aims-calc', xc='LDA')
opt = BFGS(atoms, trajectory='opt.aims.traj', logfile='opt.aims.log')

# For running with UNIX socket, put unixsocket='mysocketname'
# instead of port cf. aims parameters above
with aims.socketio(port=port) as atoms.calc:
    opt.run(fmax=0.05)
