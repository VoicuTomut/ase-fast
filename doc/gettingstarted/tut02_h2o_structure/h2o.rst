
Use different calculators and sockets
-------------------------------------

For a list of available calculators, see :mod:`ase.calculators` or run

  $ ase info --calculators

Let's run the :mol:`'H_2O'` relaxation with FHI-Aims instead.  
But in the list above, Aims
(probably) wasn't listed as available.
We first need to tell ASE how to run Aims
-- a typical configuration step for many ASE calculators.
This means specifying 1)
the command used to run Aims, and 2) where to find information about
chemical species.  We can do this by setting environment variables in the
shell:

::

   $ export ASE_AIMS_COMMAND=aims.x
   $ export AIMS_SPECIES_DIR=/home/alumne/software/FHIaims/species_defaults/light

Now ``ase info --calculators`` should tell us that it thinks
Aims is installed as ``aims.x``.

However, if we open a new shell it will forget this.  And we don't want to
modify ``.bashrc`` on these computers.  Let's instead set these variables
in our Python script:

::

   import os
   os.environ['ASE_AIMS_COMMAND'] = 'aims.x'
   os.environ['AIMS_SPECIES_DIR'] = '/home/alumne/software/FHIaims/species_defaults/light'


.. admonition:: Exercise

  Run a structure optimization of :mol:`H_2O`
  using the FHI-:class:`~ase.calculators.aims.Aims` calculator.

To enable the calculation of forces, you will need ``compute_forces=True``.
Aims will want an explicitly given XC functional, so we put ``xc='LDA'``.
The ``xc`` keyword is supported by several ASE calculators to make it easier
to specify common XC functionals.

After running the calculation, some new files will be present.
ASE has generated :file:`control.in` and :file:`geometry.in`, then
ran FHI-aims on them, producing :file:`aims.out`.
Be sure to briefly inspect the files.
Being perfectionist and/or paranoid,
we of course want to be sure that the ASE interface
set the parameters the way we wanted them.

Most ASE calculators can be made to generate a file
without triggering a calculation using ``calc.write_input_file(atoms)``.
This is useful, say, if you want to generate the files now but run them
later, with or without ASE.

ASE knows many file formats.  :func:`ase.io.read` can read both the
input file and the output file, returning :class:`~ase.Atoms`.
These files can also be opened directly with the ASE GUI.

Note that by default, subsequent calculations will overwrite each other.
Hence the Aims input and output files correspond to the final step of the
structure relaxation.  The documentation on :mod:`ase.optimize`
will tell us that we can override this behaviour by adding an observer,
or using the even more flexible :meth:`ase.optimize.Dynamics.irun` method
to force different steps into different directories.

Appendix: Communication between calculators and codes
-----------------------------------------------------

What follows is not necessary knowledge for normal usage of ASE. Unless
you are interested in how to optimize the communication between ASE and
external calculators you may skip ahead.

Different calculators communicate with computational codes in different ways.
GPAW is written in Python, so ASE and GPAW run within the same process.
However FHI-aims is a separate program.  What the Aims calculator
does for us is to generate an input file, run FHI-aims, read the output,
and return the results.

We just ran a relaxation which involved multiple geometry steps.  Each
step, a new Aims process is started and later stopped.  This is
inefficient because the ground-state density and wavefunctions of one
step would be an excellent initial guess for the next step, lowering
the number of steps necessary to converge.
But these quantities are lost when the program terminates.  To get
the best performance in structure optimisations and dynamics, we need to
avoid this loss of efficiency.

Many ASE calculators support more advanced ways of communicating.
These calculators can communicate with persistent external processes
over pipes (:class:`~ase.calculators.lammpsrun.Lammpsrun`, :class:`~ase.calculators.cp2k.CP2K`) or sockets (:class:`~ase.calculators.siesta.Siesta`,
:class:`~ase.calculators.aims.Aims`, :class:`~ase.calculators.espresso.Espresso`),
or they can work within the same process
through direct library calls
(:class:`~ase.calculators.lammpslib.Lammpslib`, GPAW).


ASE can communicate with FHI-aims over sockets using the i-PI protocol (http://ipi-code.org/).  This is done by wrapping the calculator in a
:class:`ase.calculators.socketio.SocketIOCalculator`.  The socket
calculator will use the calculator it wraps to launch a calculation,
then run it.

The documentation on the socket I/O calculator already provides full examples,
so we only need minor adjustments to run them on our local machine.

.. admonition:: Optional exercise

   Based on our previous relaxation with FHI-aims, write a script
   which runs the same calculation using the
   :class:`ase.calculators.socketio.SocketIOCalculator`.

You can run :command:`time python3 myscript.py` to see how long time
the calculation takes in total.  How much of a speedup do you get from
running the relaxation over a socket?  INET sockets often have high
latency.  If you don't see much of a speedup, this is probably why.
In that case, try switching to a UNIX socket.

The socket I/O calculator automatically generated an input file and also
immediately launched the calculation.  Since it only launches the process
once, subsequent steps don't overwrite each other and we can find all the
intermediate steps in :file:`aims.out`.

Solutions
---------

FHI-aims optimisation:

.. literalinclude:: solution/optimise_aims.py


FHI-aims/socket-io optimisation:

.. literalinclude:: solution/optimise_aims_socketio.py
