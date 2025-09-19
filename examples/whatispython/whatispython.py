""".. _pythonintroduction:

What is Python?
===============

This section will give a very brief introduction to the Python language.

.. tip::

   See also the `Python home page <https://www.python.org/>`_ for further
   information.

Executing Python code
=====================
You can execute Python code interactively by starting the interpreter like with
the command ``python3`` and test it with any python command such as:

"""

# %%
print('hello')


# %%
# You can also put the ``print("hello")`` line in a file (``hello.py``) and
# execute it as a Python script with ``python3 hello.py`` or ``python3 -i
# hello.py`` to enter interactive mode after the file is executed.
#
# Finally, you can put ``#!/usr/bin/env python3`` in the first line of the
# ``hello.py`` file, make it executable (``chmod +x hello.py``) and execute it
# like any other executable.
#
#
#
# .. tip::
#
#    For a better interactive experience, consider ipython.
#
# Types
# =====
#
# .. list-table:: Supported Python Types
#    :header-rows: 1
#    :widths: 15 30 30
#
#    * - **Type**
#      - **Description**
#      - **Example**
#    * - ``bool``
#      - Boolean
#      - ``False``
#    * - ``int``
#      - Integer
#      - ``117``
#    * - ``float``
#      - Floating point number
#      - ``1.78``
#    * - ``complex``
#      - Complex number
#      - ``0.5 + 2.0j``
#    * - ``str``
#      - String
#      - ``'abc'``
#    * - ``tuple``
#      - Tuple
#      - ``(1, 'hmm', 2.0)``
#    * - ``list``
#      - List
#      - ``[1, 'hmm', 2.0]``
#    * - ``dict``
#      - Dictionary
#      - ``{'a': 7.0, 23: True}``
#
# A dict object is mapping from keys to values:

# %%
d = {'s': 0, 'p': 1}
d['d'] = 2
print('the whole dictionary:', d)
print('one entry of the dictionary:', d['p'])

# %%
# In this example all keys are strings and all values are integers. Types can
# be freely mixed in the same dictionary; any type can be used as a value and
# most types can be used as keys (mutable objects cannot be keys).
#
# A ``list`` object is an ordered collection of arbitrary objects:

# %%
l = [1, ('gg', 7), 'hmm', 1.2]
print('the whole list:', l)
print('one list element:', l[1])
print('negative index:', l[-2])

# %%
# Indexing a list with negative numbers counts from the end of the list, so
# element ``-2`` is the second last.
#
# A tuple behaves like a list -- except that it can’t be modified in place.
# Objects of types list and dict are mutable -- all the other types listed in
# the table are immutable, which means that once an object has been created, it
# can not change. Tuples can therefore be used as dictionary keys, lists
# cannot.
#
# .. note::
#
#    List and dictionary objects can change. Variables in Python are references
#    to objects -- think of the ``=`` operator as a “naming operator”, not as
#    an assignment operator. This is demonstrated here:
#

# %%
a = ['q', 'w']
b = a
a.append('e')
print('the original, changed list:', a)
print('the second list:', b)

# %%
# The line ``b = a`` gives a new name to the array, and both names now refer to
# the same list.
#
# However, often a new object is created and named at the same time, in this
# example the number ``42`` is not modified, a new number ``47`` is created and
# given the name ``d``. And later, ``e`` is a name for the number ``47``, but
# then a new number ``48`` is created, and ``e`` now refers to that number:

# %%
c = 42
d = c + 5
print('the first number:', c)
print('the second number:', d)

e = d
e += 1
print('second and third number:', (d, e))

# %%
# .. note::
#
#    Another very important type is the ``ndarray`` type described here:
#    `Numeric arrays in Python <https://ase-lib.org/numpy.html#numpy>`_. It is
#    an array type for efficient numerics, and is heavily used in ASE.
#
# Loops
# =====
# A loop in Python can be done like this:

# %%
things = ['a', 7]
for x in things:
    print(x)

# %%
# The ``things`` object could be any sequence. Strings, tuples, lists,
# dictionaries, ndarrays and files are sequences. Try looping over some of
# these types.
#
# Often you need to loop over a range of numbers:

# %%
for i in range(5):
    print(i, i * i)

# %%
# Functions and classes
# =====================
#
# A function is defined like this:


# %%
def f(x, m=2, n=1):
    y = x + n
    return y**m


print(f(5))
print(f(5, n=8))

# %%
# Here ``f`` is a function, ``x`` is an argument, ``m`` and ``n`` are keywords
# with default values ``2`` and ``1`` and ``y`` is a variable.
#
# A class is defined like this:


# %%
class A:
    def __init__(self, b):
        self.c = b

    def m(self, x):
        return self.c * x

    def get_c(self):
        return self.c


# %%
# You can think of a class as a template for creating user defined objects. The
# ``__init__()`` function is called a constructor, it is being called when
# objects of this type are being created.
#
# In the class ``A`` ``__init__`` is a constructor, ``c`` is an attribute and
# ``m`` and ``get_c`` are methods.

# %%
a = A(7)
print(a.c)
print(a.get_c())
print(a.m(3))

# %%
# Here we make an instance (or object) ``a`` of type ``A``.
#
# Importing modules
# =================
#
# There are several ways to import modules (either ``.py`` files in your
# working directory or packages that are installed in your python environment).
# For example:
#
# ::
#
#   import numpy; numpy.linspace(1, 10, num=5)
#
# and
#
# ::
#
#   import numpy as np; np.linspace(1, 10, num=5)
#
# and
#
# ::
#
#   from numpy import linspace; linspace(1, 10, num=5)
#
# all yield the same result.
