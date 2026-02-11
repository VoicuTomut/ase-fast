.. _addingexamples:

=====================
How to add an example
=====================

Contributions to ASE are very welcome and appreciated.


GitLab repository
=================
Our examples are all part of the main ASE Gitlab_ repository.

.. _Gitlab: https://gitlab.com/ase/ase

Please read and follow our contribution guidelines (:ref:`contribute`).

You can find the source code of existing examples in the :git:`examples` folder in Gitlab_.
These are rendered as the :ref:`tutorials` on the ASE website.

.. _examples: https://gitlab.com/ase/ase/-/tree/master/examples

Sphinx-Gallery
==============
We are using Sphinx-Gallery to test and display our examples.
Every time a change is proposed on Gitlab (to the code or the documentation), all examples are
tested and errors are flagged.

Tutorial Guidelines
===================
When contributing a new example, it should roughly have the following structure:

- Learning outcomes and Objectives: describe the task and the aim of the tutorial
- Imports
- Setup: load necessary data
- Task description: describe the basic functionality of used functions and add links to the ASE documentation where needed
- Basic working example
- Further information: links, literature, references to the documentation


Adding an Example
=================
You need to following to add an example:

- a clone of your fork of the ase gitlab
- a python environment with relevant packages installed. These are defined in the ase "docs" dependency group and can be installed with

.. code-block:: console

    $ pip install ase[docs]

Location of Examples
--------------------
All examples are located in the :git:`examples` folder. More advanced ASE tutorials
can be found in the folder ``examples/tutorials``, which is most likely where 
you want to add your contribution. There is also an ``examples/python`` folder
for python related tutorials.


Format of Examples
------------------
In the folder, all examples are collected as ``.py`` files. These ``.py`` files
must contain a title in reStructuredText_.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

.. Note::

   Please do not use any non-human written files such as .png files in the tutorials, generate them during the tutorial if possible. Also, do not add large files (> few 10s of Kb). Sphinx-gallery creates a downloadable ``.zip`` folder for all examples, so we want to keep it light. 

Adding a ``.py`` file to the example folder will automatically make sphinx-gallery find it and it will be executed when building the gallery.


.. note:

   When moving one of the old tutorials/examples to a sphinx-gallery example, please add a deprecation note and a link to the new example. 

Linting
^^^^^^^
We use a linter that checks all the examples commited. Please check your example before commiting and apply the changes necessary so it passes a linter without complains.
Your merge request will not be accepted unless all tests passed, which also includes linting.
The linter we are using for our CI check is Ruff_.

.. _Ruff: https://github.com/astral-sh/ruff

You can simply install it with pip:

.. code-block:: console

    $ pip install ruff

Before opening a merge request for your example, please make sure that these two checks are passed when executed in the ase folder:

.. code-block:: console

    $ ruff check

.. code-block:: console

    $ ruff format

Building the Gallery
--------------------
You can build the gallery from the 'doc' directory.
For this, you can run the following two commands from the terminal:

.. code-block:: console

    $ make html

.. code-block:: console

    $ make browse





