# %%
"""
.. _intro_ase_db:

Introduction to ASE databases
=============================
ASE has its own database format that can be used for storing and retrieving
atoms (and associated data) in a compact and convenient way. In the following,
we will create databases and interact with them through python scripts and
the command line.

Setting up a database
---------------------
To construct a database we first need some atomic structures so let's quickly
create some. As you have seen the ASE command line tool provides many convenient
commands and in particular we can use the ``build`` command to create some
atomic structures. Remember, if you are unsure how to use a particular command
you can always append ``-h`` to the particular command (ie. ``ase build -h``)
to see the help for that particular command.

We choose to build silicon, germanium and carbon in the diamond crystal
structure for which ASE already knows the lattice constants:

::

  $ ase build -x diamond Si
  $ ase build -x diamond Ge
  $ ase build -x diamond C

This creates three files: :file:`Si.json`, :file:`Ge.json` and :file:`C.json`.
If you want to, you can inspect them with ASE's ``gui`` command, however we
want to construct a database containing these structures. To do this we can use
``convert``::

  $ ase convert Si.json C.json Ge.json database.db

This has created an ASE database name :file:`database.db`.

Additionally, one can use the ``ase build`` command to build Si, Ge and C and
convert them into a ASE database named :file:`database.db` using the following:
"""

from pathlib import Path

from gpaw import GPAW, PW

from ase.build import bulk
from ase.constraints import ExpCellFilter
from ase.db import connect
from ase.dft.bandgap import bandgap
from ase.optimize import BFGS

dbfile = Path('database.db')
dbfile.unlink(missing_ok=True)

structures = ['Si', 'Ge', 'C']
db = connect('database.db')

for f in structures:
    db.write(bulk(f))

# %%
# Inspecting a database on the command line
# -----------------------------------------
# We can inspect the database using the ``db`` command::

# %%
#  $ ase db database.db

# %%
# which will display three entries, one for each structure. From this point
# it is advised to bring up the help for the ``db`` command every time you need
# it.
#
# From the help we can see that it is possible to make selections (queries in
# database lingo) in the database by::
#
#    $ ase db database.db Si
#
# which will show all structures containing silicon. To see the details of a
# particular row we can do::
#
#  $ ase db database.db Si -l
#
# From which we can get an overview of the stored data. We can also view all
# structures in a database using::
#
#  $ ase gui database.db
#
# or if we want to view a single one we can do::
#
#  $ ase gui database.db@Si
#
# where everything after the @ is interpreted as a query.


# %%
# Opening a database using a Python script
# ----------------------------------------
# To open a database using Python, we can use the :class:`ase.database.connect`
#  method which returns a database object from which we can make selections::
#
#  from ase.db import connect
#
#  db = connect('database.db')
#  for row in db.select():
#      atoms = row.toatoms()
#      print(atoms)
#
#
# We can make selections in the database using ``db.select(some_selection)``
# which returns all rows matching ``some_selection``. In this case
# ``some_selection`` was omitted which means that we select all rows in
# the database. For each row the associated :class:`ase.Atoms` objects
# is retrieved by using the ``row.toatoms()`` method.
#
# .. admonition:: Hint
#
#   In order to see the documentation for a particular
#   python function import it and use the ``help`` function.
#   For example
#   ::
#
#     from ase.db import connect
#     db = connect('database.db')
#     help(db.select)
#
#
#   will show the documentation for the select method of the database
#   object. Another useful function is ``dir`` which shows
#   all attributes of a python object. For example
#   ::
#
#    from ase.db import connect
#    db = connect('database.db')
#    row = db.select(id=1)[0]
#    dir(row)
#
#   will show all attributes of the row object.
#
#
#   Using a python script, print the formula for each row in your database.

# %%
# Write new entries to a database using Python
# --------------------------------------------
# Next, we loop through all materials, relax them
# (see exercise "Structure Optimization")
# and save the relaxed structure as a new entry in the database with an
# added column relaxed equal to ``True`` that we can use later for selecting
# only these materials. A new entry in the database can be written using the
# ``write()`` method of a database object.

# %%
#  CAUTION: To relax crystals we have to specify that the cell parameters
#  should be relaxed as well. This is done by wrapping
#  :class:`ase.constraints.ExpCellFilter` around the atoms object like::
#
#      filter = ExpCellFilter(atoms)
#
#   and feeding ``filter`` into the optimization routine see
#   ``help(ExpCellFilter)`` for more explanation.

for row in db.select():
    atoms = row.toatoms()
    calc = GPAW(
        mode=PW(400), kpts=(4, 4, 4), txt=f'{row.formula}-gpaw.txt', xc='LDA'
    )
    atoms.calc = calc
    atoms.get_stress()
    filter = ExpCellFilter(atoms)
    opt = BFGS(filter)
    opt.run(fmax=0.05)
    db.write(atoms=atoms, relaxed=True)


# %%
# Adding data to existing database
# --------------------------------
# Now we want to calculate some data and include the data in the database
# which can be done using the ``update`` method of the database object.
# To do so, we loop through all materials in the database and make a
# self consistent calculation using GPAW in plane wave mode for all materials.
# Then use the :class:`ase.dft.bandgap.bandgap()` method to calculate the
# bandgap of the materials and store it under the ``bandgap`` keyword.


for row in db.select(relaxed=True):
    atoms = row.toatoms()
    calc = GPAW(
        mode=PW(400), kpts=(4, 4, 4), txt=f'{row.formula}-gpaw.txt', xc='LDA'
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    bg, _, _ = bandgap(calc=atoms.calc)
    db.update(row.id, bandgap=bg)

# %%
# Now, we can inspect the  database again using the
# ``ase db`` command. To see the new column ``bandgap`` you can display all
# columns using the ``-c++`` option::
#
#  $ ase db database -c++

# %%
# Browsing data
# -------------
# The database can also be visualized in a browser by using::
#
#  $ ase database database.db -w
#  $ firefox http://0.0.0.0:5000/
#
# This opens a local webserver which can be opened in firefox like above. The
# layout can be customized further than our simple example however this would
# probably be too much for now. To see a more advanced example of such a web
# interfaced database in action you can check out the 2D database
# https://cmrdb.fysik.dtu.dk/c2db.

# %%
# Advanced tutorial
# -----------------
# An additional tutorial using the ASE databases for adsorbates on metals
# can be found at https://ase-lib.org/tutorials/db/db.html.
