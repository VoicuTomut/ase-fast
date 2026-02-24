""".. _numpy:

========================
Numeric arrays in Python
========================

Examples demonstrating NumPy usage in ASE documentation.

Links to NumPy's webpage:

* `Numpy and Scipy Documentation`_
* `Numpy user guide <https://docs.scipy.org/doc/numpy/user/index.html>`_


.. _Numpy and Scipy Documentation: https://docs.scipy.org/doc/

ASE makes heavy use of an extension to Python called NumPy.  The
NumPy module defines an ``ndarray`` type that can hold large arrays of
uniform multidimensional numeric data.  An array is similar to a
``list`` or a ``tuple``, but it is a lot more powerful and efficient.

Some examples from everyday ASE-life here ...
"""

import numpy as np

a = np.zeros((3, 2))
a[:, 1] = 1.0
a[1] = 2.0
print(a)
print(a.shape)
print(a.ndim)

# %%
# The conventions of numpy's linear algebra package:

# --- Basic array example ---
a = np.zeros((3, 2))
a[:, 1] = 1.0
a[1] = 2.0
print('Array a:\n', a)
print('Shape:', a.shape)
print('Number of dimensions:', a.ndim)

# --- Linear algebra example ---
# Make a random Hermitian matrix
H = np.random.rand(6, 6) + 1.0j * np.random.rand(6, 6)
H = H + H.T.conj()

# Eigenvalues and eigenvectors
eps, U = np.linalg.eigh(H)

# Sort eigenvalues and corresponding eigenvectors
sorted_indices = eps.real.argsort()
eps = eps[sorted_indices]
U = U[:, sorted_indices]

# Verify diagonalization
print(
    'Check diagonalization:\n', np.dot(np.dot(U.T.conj(), H), U) - np.diag(eps)
)
print('All close?', np.allclose(np.dot(np.dot(U.T.conj(), H), U), np.diag(eps)))

# Check eigenvectors individually
print(
    'Eigenvector check (one column):',
    np.allclose(np.dot(H, U[:, 3]), eps[3] * U[:, 3]),
)
print('Eigenvector check (all):', np.allclose(np.dot(H, U), eps * U))

# %%
# The rules for multiplying 1D arrays with 2D arrays:
#
# * 1D arrays and treated like shape (1, N) arrays (row vectors).
# * left and right multiplications are treated identically.
# * A length :math:`m` *row* vector can be multiplied with an :math:`n \times m`
#   matrix, producing the same result as if replaced by a matrix with
#   :math:`n` copies of the vector as rows.
# * A length :math:`n` *column* vector can be multiplied with
#   an :math:`n \times m`
#   matrix, producing the same result as if replaced by a matrix with
#   :math:`m` copies of the vector as columns.
#
# Thus, for the arrays below:

# --- 1D vs 2D multiplication rules ---
M = np.arange(5 * 6).reshape(5, 6)  # A matrix of shape (5, 6)
v5 = np.arange(5) + 10  # A vector of length 5
v51 = v5[:, None]  # Column vector (5, 1)
v6 = np.arange(6) - 12  # A vector of length 6
v16 = v6[None, :]  # Row vector (1, 6)

# %%
# The following identities hold::
#
#   v6 * M == v16 * M == M * v6 == M * v16 == M * v16.repeat(5, 0)
#   v51 * M == M * v51 == M * v51.repeat(6, 1)
#
# The same rules apply for adding and subtracting 1D arrays to
# from 2D arrays.

# Identities
print('v6 * M == M * v6?', np.allclose(v6 * M, M * v6))
print('v16 * M == M * v16?', np.allclose(v16 * M, M * v16))
print('v51 * M == M * v51?', np.allclose(v51 * M, M * v51))
