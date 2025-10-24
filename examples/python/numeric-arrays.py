""" .. _python_numeric_arrays:

========================
Numeric arrays in Python
========================

Examples demonstrating NumPy usage in ASE documentation.
"""

import numpy as np

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

# --- 1D vs 2D multiplication rules ---
M = np.arange(5 * 6).reshape(5, 6)  # A matrix of shape (5, 6)
v5 = np.arange(5) + 10  # A vector of length 5
v51 = v5[:, None]  # Column vector (5, 1)
v6 = np.arange(6) - 12  # A vector of length 6
v16 = v6[None, :]  # Row vector (1, 6)

# Identities
print('v6 * M == M * v6?', np.allclose(v6 * M, M * v6))
print('v16 * M == M * v16?', np.allclose(v16 * M, M * v16))
print('v51 * M == M * v51?', np.allclose(v51 * M, M * v51))
