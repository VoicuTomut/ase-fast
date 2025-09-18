.. _addingexamples:

=====================
How to add an example
=====================

Contributions to ASE are very welcome and appreciated.


GitLab repository
=================
Our examples are all part of the main ASE Gitlab_ repository.

.. _Gitlab: https://gitlab.com/ase/ase

Please read and follow our contribution guidelines (:ref: contribute:).

On Gitlab you can find existing examples in the examples folder.


Sphinx-Gallery
==============
We are using Sphinx-Gallery to test and display our examples.
Every time a new example is added in Gitlab, all examples are
tested and errors are flagged.

Tutorial Guidelines
===================
When contributing a new example, it should roughly have the following structure:

- Introduction: describe the task and the aim of the tutorial
- Imports
- Setup: load necessary data
- Learning outcomes: describe the basic functionality of used functions and add links to the ASE documentation where needed
- Basic working example
- Further information: links, literature, references to the documentation


Adding an Example
=================
You need to following to add an example:

- a clone of your fork of the ase gitlab
- a python environment with sphinx-gallery and sphinx-box-theme installed. You can install this with


.. code-block:: console

    $ pip install sphinx_book_theme --user

.. code-block:: console

    $ pip install sphinx-gallery --user

Location of Examples
--------------------
All examples are located in the ``examples`` folder. More advanced ASE tutorials
can be found in the folder ``examples/tutorials``, which is most likely where 
you want to add your contribution. There is also a ``examples/python`` folder
for python related tutorials.


Format of Examples
------------------
In the folder, all examples are collected as ``.py`` files. These ``.py`` files
must contain a title in reStructuredText_.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

.. Note::

   Please do not use any non-human written files such as .png files in the tutorials, generate them from scratch if possible. Also, do not add large files (> few 10s of Kb). Sphinx-gallery creates a downloadable ``.zip`` folder for all examples, so we want to keep it light. 

Adding a ``.py`` file to the example folder will automatically make sphinx-gallery find it and it will be executed when building the gallery.


.. note:

   When moving one of the old tutorials/examples to a sphinx-gallery example, please add a deprecation note and a link to the new example. 


Building the Gallery
--------------------
You can build the gallery from the 'doc' directory.
For this, you can run the following two commands from the terminal:

.. code-block:: console

    $ make html

.. code-block:: console

    $ make browse





