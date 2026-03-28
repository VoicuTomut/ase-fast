use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{IntoPyArray, PyReadonlyArray2};

mod writer;
mod reader;

// ─── Write path ─────────────────────────────────────────────────────────────

/// Build the complete extxyz atom block (all `natoms` lines) as a Python str.
///
/// Replaces the Python loop `for i in range(natoms): fileobj.write(fmt % tuple(data[i]))`.
///
/// `sym_codepoints` is a (natoms, max_str_len) uint32 array of UTF-32 LE codepoints,
/// obtained from `symbols_array.view(np.uint32).reshape(natoms, -1)`.  Rust decodes
/// these in-place without creating Python string objects, avoiding O(N) Python overhead.
///
/// Column layout is described by `col_types` (list of ints: ord('s'), ord('f'), etc.):
///   ord('s') → next symbol from sym_codepoints (left-justified to `sym_width`)
///   ord('f') → next column from `float_mat`  (%16.8f)
///   ord('i') → next column from `int_mat`    (%8d)
///   ord('b') → next column from `bool_mat`   (' %.1s' → "  T" / "  F")
///
/// `float_mat`, `int_mat`, `bool_mat` may have 0 columns.
#[pyfunction]
fn write_atoms_rs<'py>(
    _py: Python<'py>,
    sym_codepoints: PyReadonlyArray2<'py, u32>,
    sym_width: usize,
    float_mat: PyReadonlyArray2<'py, f64>,
    int_mat: PyReadonlyArray2<'py, i64>,
    bool_mat: PyReadonlyArray2<'py, u8>,
    col_types: Vec<u8>,
) -> PyResult<String> {
    Ok(writer::format_atom_block_codepoints(
        sym_codepoints.as_array(),
        sym_width,
        float_mat.as_array(),
        int_mat.as_array(),
        bool_mat.as_array(),
        &col_types,
    ))
}

// ─── Read path ───────────────────────────────────────────────────────────────

/// Parse `natoms` atom lines, returning a Python dict with keys:
///   'floats' → ndarray shape (natoms, n_float_cols), dtype float64
///   'ints'   → ndarray shape (natoms, n_int_cols),   dtype int64
///   'bools'  → ndarray shape (natoms, n_bool_cols),  dtype uint8 (0/1)
///   'strs'   → Python list of lists of str, one sub-list per string column
///
/// `col_types` is a bytes/list of ints, one per token per line:
///   b'f'=102 → float, b'i'=105 → int, b's'=115 → string, b'b'=98 → bool
///
/// Raises `ValueError` if any line cannot be parsed according to `col_types`.
#[pyfunction]
fn read_atom_lines_rs<'py>(
    py: Python<'py>,
    lines: Vec<String>,
    col_types: Vec<u8>,
) -> PyResult<Bound<'py, PyDict>> {
    let natoms = lines.len();

    let parsed = reader::parse_atom_lines(&lines, &col_types)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let dict = PyDict::new(py);

    // floats → (natoms, n_float_cols) ndarray
    {
        let arr = if parsed.n_float_cols > 0 {
            let shape = [natoms, parsed.n_float_cols];
            ndarray::Array2::from_shape_vec(shape, parsed.floats)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            ndarray::Array2::<f64>::zeros((natoms, 0))
        };
        dict.set_item("floats", arr.into_pyarray(py))?;
    }

    // ints → (natoms, n_int_cols) ndarray
    {
        let arr = if parsed.n_int_cols > 0 {
            let shape = [natoms, parsed.n_int_cols];
            ndarray::Array2::from_shape_vec(shape, parsed.ints)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            ndarray::Array2::<i64>::zeros((natoms, 0))
        };
        dict.set_item("ints", arr.into_pyarray(py))?;
    }

    // bools → (natoms, n_bool_cols) ndarray (uint8, 0/1)
    {
        let arr = if parsed.n_bool_cols > 0 {
            let shape = [natoms, parsed.n_bool_cols];
            ndarray::Array2::from_shape_vec(shape, parsed.bools)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            ndarray::Array2::<u8>::zeros((natoms, 0))
        };
        dict.set_item("bools", arr.into_pyarray(py))?;
    }

    // strs → list of lists of str
    let str_list = PyList::new(
        py,
        parsed.strs.iter().map(|col| {
            PyList::new(py, col.iter()).unwrap().into_any()
        }),
    )?;
    dict.set_item("strs", str_list)?;

    Ok(dict)
}

// ─── Module ──────────────────────────────────────────────────────────────────

#[pymodule]
fn _extxyz_rs(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_atoms_rs, m)?)?;
    m.add_function(wrap_pyfunction!(read_atom_lines_rs, m)?)?;
    Ok(())
}
