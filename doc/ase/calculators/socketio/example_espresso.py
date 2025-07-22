import sys

from ase.build import molecule
from ase.calculators.espresso import Espresso
from ase.calculators.socketio import SocketIOCalculator
from ase.optimize import BFGS

atoms = molecule('H2O', vacuum=3.0)
atoms.rattle(stdev=0.1)

# Environment-dependent parameters (please configure before running).
# Use any files you have, but these are available in the ase-datafiles project.
pseudopotentials = {'H': 'h_lda_v1.4.uspp.F.UPF', 'O': 'o_lda_v1.2.uspp.F.UPF'}

# In this example we use a UNIX socket.  See other examples for INET socket.
# UNIX sockets are faster then INET sockets, but cannot run over a network.
# UNIX sockets are files.  The actual path will become /tmp/ipi_ase_espresso.
unixsocket = 'ase_espresso'

# See also QE documentation, e.g.:
#
#    https://www.quantum-espresso.org/Doc/pw_user_guide/node13.html

espresso = Espresso(
    directory='calc-espresso',
    ecutwfc=30.0,
    pseudopotentials=pseudopotentials,
)

opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')

with espresso.socketio(unixsocket=unixsocket) as atoms.calc:
    opt.run(fmax=0.05)

# Note: QE does not generally quit cleanly - expect nonzero exit codes.
