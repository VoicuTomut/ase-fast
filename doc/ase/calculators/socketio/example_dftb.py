import sys

from ase.build import molecule
from ase.calculators.dftb import Dftb
from ase.calculators.socketio import SocketIOCalculator
from ase.optimize import BFGS

# DFTB must be compiled with socket support in order for this to work:
# "cmake -D WITH_SOCKETS=TRUE ..."
socketfile = 'Hello'

socket_kwargs = dict(
    Driver_='',
    Driver_Socket_='',
    Driver_Socket_File=socketfile,
)

atoms = molecule('H2O')
dftb = Dftb(
    directory='dftb-calc',
    Hamiltonian_MaxAngularMomentum_='',
    Hamiltonian_MaxAngularMomentum_O='"p"',
    Hamiltonian_MaxAngularMomentum_H='"s"',
    **socket_kwargs,
)

opt = BFGS(atoms, trajectory='opt.traj')

# dftb does not have a .socketio() shortcut method, so we create the socketio
# calculator manually (and must remember specify the same socket file):
with SocketIOCalculator(dftb, log=sys.stdout, unixsocket=socketfile) as calc:
    atoms.calc = calc
    opt.run(fmax=0.01)
