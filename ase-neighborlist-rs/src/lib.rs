use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

mod cell_list;

/// Compute a neighbor list for an atomic configuration (Rust fast path).
///
/// Implements the same cell-list algorithm as ASE's `primitive_neighbor_list`
/// for the common case of a scalar float cutoff.  Returns all five quantities
/// (i, j, d, D, S) always; the Python wrapper selects which to expose.
///
/// Parameters
/// ----------
/// pbc : [bool; 3]
///     Periodic boundary conditions for each axis.
/// cell : (3, 3) float64 array
///     Unit cell vectors (row-major: cell[0] = first lattice vector).
/// positions : (N, 3) float64 array
///     Cartesian or scaled atom positions.
/// cutoff : f64
///     Neighbour cutoff radius in the same units as positions.
/// self_interaction : bool
///     Include (i, i, 0) pairs.
/// use_scaled_positions : bool
///     If true, `positions` are fractional coordinates.
/// max_nbins : usize
///     Maximum total number of spatial bins (memory cap).
#[pyfunction]
#[pyo3(signature = (pbc, cell, positions, cutoff, self_interaction=false, use_scaled_positions=false, max_nbins=1_000_000))]
fn primitive_neighbor_list_rs<'py>(
    py: Python<'py>,
    pbc: [bool; 3],
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let cell = cell.as_array();
    let positions = positions.as_array();

    let (i_arr, j_arr, d_arr, big_d, s_arr) = cell_list::build_neighbor_list(
        &pbc,
        &cell,
        &positions,
        cutoff,
        self_interaction,
        use_scaled_positions,
        max_nbins,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    Ok((
        i_arr.into_pyarray(py).into(),
        j_arr.into_pyarray(py).into(),
        d_arr.into_pyarray(py).into(),
        big_d.into_pyarray(py).into(),
        s_arr.into_pyarray(py).into(),
    ))
}

/// Neighbor list with per-atom radii cutoffs.
///
/// Equivalent to ASE's `primitive_neighbor_list` when `cutoff` is a 1-D array
/// of per-atom radii.  Atoms i,j are neighbors when d < radii[i] + radii[j].
///
/// Uses the scalar fast path with max_cutoff = 2*max(radii) and post-filters.
#[pyfunction]
#[pyo3(signature = (pbc, cell, positions, radii, self_interaction=false, use_scaled_positions=false, max_nbins=1_000_000))]
fn primitive_neighbor_list_radii_rs<'py>(
    py: Python<'py>,
    pbc: [bool; 3],
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let cell = cell.as_array();
    let positions = positions.as_array();
    let radii = radii.as_array();

    let (i_arr, j_arr, d_arr, big_d, s_arr) = cell_list::build_neighbor_list_radii(
        &pbc,
        &cell,
        &positions,
        &radii,
        self_interaction,
        use_scaled_positions,
        max_nbins,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    Ok((
        i_arr.into_pyarray(py).into(),
        j_arr.into_pyarray(py).into(),
        d_arr.into_pyarray(py).into(),
        big_d.into_pyarray(py).into(),
        s_arr.into_pyarray(py).into(),
    ))
}

/// Neighbor list with element-pair (dict-style) cutoffs.
///
/// Equivalent to ASE's `primitive_neighbor_list` when `cutoff` is a dict
/// mapping `(Z_i, Z_j)` atomic-number pairs to cutoff distances.
///
/// Arguments
/// ---------
/// numbers  : (N,) int64 — atomic numbers per atom
/// zi_keys  : cutoff dict keys (first element atomic number)
/// zj_keys  : cutoff dict keys (second element atomic number)
/// cutoff_vals : corresponding cutoff distances
///
/// The lookup is symmetric: (Z_i, Z_j) and (Z_j, Z_i) are treated equally.
#[pyfunction]
#[pyo3(signature = (pbc, cell, positions, numbers, zi_keys, zj_keys, cutoff_vals, self_interaction=false, use_scaled_positions=false, max_nbins=1_000_000))]
fn primitive_neighbor_list_dict_rs<'py>(
    py: Python<'py>,
    pbc: [bool; 3],
    cell: PyReadonlyArray2<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
    numbers: PyReadonlyArray1<'py, i64>,
    zi_keys: Vec<i64>,
    zj_keys: Vec<i64>,
    cutoff_vals: Vec<f64>,
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let cell = cell.as_array();
    let positions = positions.as_array();
    let numbers = numbers.as_array();

    let (i_arr, j_arr, d_arr, big_d, s_arr) = cell_list::build_neighbor_list_dict(
        &pbc,
        &cell,
        &positions,
        &numbers,
        &zi_keys,
        &zj_keys,
        &cutoff_vals,
        self_interaction,
        use_scaled_positions,
        max_nbins,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    Ok((
        i_arr.into_pyarray(py).into(),
        j_arr.into_pyarray(py).into(),
        d_arr.into_pyarray(py).into(),
        big_d.into_pyarray(py).into(),
        s_arr.into_pyarray(py).into(),
    ))
}

#[pymodule]
fn _neighborlist_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(primitive_neighbor_list_rs, m)?)?;
    m.add_function(wrap_pyfunction!(primitive_neighbor_list_radii_rs, m)?)?;
    m.add_function(wrap_pyfunction!(primitive_neighbor_list_dict_rs, m)?)?;
    Ok(())
}
