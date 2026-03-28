//! Supercell position builder — Rust port of the hot loop in
//! ase/build/supercells.py::make_supercell().
//!
//! Only the position broadcast is Rust-accelerated; cell properties,
//! validation, and array copying are still in Python.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Build Cartesian positions for a supercell.
///
/// Equivalent to the NumPy broadcast in make_supercell():
///   cell-major:  shifted[l, a, :] = prim_positions[a, :] + lattice_points[l, :]
///                → output[(l * N_prim + a), :] = prim[a] + lp[l]
///   atom-major:  shifted[a, l, :] = prim_positions[a, :] + lattice_points[l, :]
///                → output[(a * N_super + l), :] = prim[a] + lp[l]
///
/// Parameters
/// ----------
/// prim_positions : (N_prim, 3) float64 — atom positions in the primitive cell
/// lattice_points : (N_super, 3) float64 — Cartesian lattice point offsets
/// cell_major     : bool — if True, cell-major ordering; else atom-major
///
/// Returns
/// -------
/// (N_prim * N_super, 3) float64 array of Cartesian positions.
#[pyfunction]
pub fn make_supercell_positions_rs<'py>(
    py: Python<'py>,
    prim_positions: PyReadonlyArray2<'py, f64>,
    lattice_points: PyReadonlyArray2<'py, f64>,
    cell_major: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let prim = prim_positions.as_array();
    let lpts = lattice_points.as_array();

    let n_prim  = prim.nrows();
    let n_super = lpts.nrows();
    let n_total = n_prim * n_super;

    let mut out = Array2::<f64>::zeros((n_total, 3));

    if cell_major {
        // output[l * n_prim + a, :] = prim[a, :] + lpts[l, :]
        for l in 0..n_super {
            for a in 0..n_prim {
                let row = l * n_prim + a;
                out[[row, 0]] = prim[[a, 0]] + lpts[[l, 0]];
                out[[row, 1]] = prim[[a, 1]] + lpts[[l, 1]];
                out[[row, 2]] = prim[[a, 2]] + lpts[[l, 2]];
            }
        }
    } else {
        // atom-major: output[a * n_super + l, :] = prim[a, :] + lpts[l, :]
        for a in 0..n_prim {
            for l in 0..n_super {
                let row = a * n_super + l;
                out[[row, 0]] = prim[[a, 0]] + lpts[[l, 0]];
                out[[row, 1]] = prim[[a, 1]] + lpts[[l, 1]];
                out[[row, 2]] = prim[[a, 2]] + lpts[[l, 2]];
            }
        }
    }

    Ok(out.into_pyarray(py).into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cell_major_two_atoms_two_points() {
        // prim: 2 atoms at [0,0,0] and [1,0,0]
        // lp: 2 lattice points at [0,0,0] and [5,0,0]
        // cell-major output order: (l=0,a=0), (l=0,a=1), (l=1,a=0), (l=1,a=1)
        let prim = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let lpts = array![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let n_prim = prim.nrows();
        let n_super = lpts.nrows();
        let n_total = n_prim * n_super;
        let mut out = Array2::<f64>::zeros((n_total, 3));
        for l in 0..n_super {
            for a in 0..n_prim {
                let row = l * n_prim + a;
                for c in 0..3 { out[[row, c]] = prim[[a, c]] + lpts[[l, c]]; }
            }
        }
        assert!((out[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((out[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((out[[2, 0]] - 5.0).abs() < 1e-10);
        assert!((out[[3, 0]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_atom_major_two_atoms_two_points() {
        // atom-major output: (a=0,l=0), (a=0,l=1), (a=1,l=0), (a=1,l=1)
        let n_prim = 2;
        let n_super = 2;
        let prim = [[0.0f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let lpts = [[0.0f64, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let mut out = Array2::<f64>::zeros((4, 3));
        for a in 0..n_prim {
            for l in 0..n_super {
                let row = a * n_super + l;
                for c in 0..3 { out[[row, c]] = prim[a][c] + lpts[l][c]; }
            }
        }
        assert!((out[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((out[[1, 0]] - 5.0).abs() < 1e-10);
        assert!((out[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((out[[3, 0]] - 6.0).abs() < 1e-10);
    }
}
