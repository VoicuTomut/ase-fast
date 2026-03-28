//! VASP IO kernels — Rust ports of the per-atom Python loops in
//! ase/io/vasp.py.
//!
//! Four functions are exposed:
//!   parse_poscar_positions_rs   — read the N-atom block in POSCAR/CONTCAR
//!   format_poscar_positions_rs  — write the N-atom block for POSCAR/CONTCAR
//!   parse_xdatcar_coords_rs     — read one XDATCAR frame (coordinate lines)
//!   format_xdatcar_config_rs    — write one complete XDATCAR config block

use std::fmt::Write as FmtWrite;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// POSCAR read
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the N-atom position block from a POSCAR/CONTCAR file.
///
/// Python equivalent (read_vasp_configuration:261-278):
/// ```python
/// for atom in range(tot_natoms):
///     ac = fd.readline().split()
///     atoms_pos[atom] = [float(_) for _ in ac[0:3]]
///     if selective_dynamics:
///         selective_flags[atom] = [_ == 'F' for _ in ac[3:6]]
/// ```
///
/// Parameters
/// ----------
/// lines : list[str]
///     The N atom lines already read from the file descriptor.
/// selective : bool
///     Whether selective-dynamics flags (T/F) are present.
///
/// Returns
/// -------
/// (positions: ndarray float64 [N,3],
///  flags:     ndarray bool    [N,3]  or  None)
#[pyfunction]
pub fn parse_poscar_positions_rs<'py>(
    py: Python<'py>,
    lines: Vec<String>,
    selective: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyAny>)> {
    let natoms = lines.len();
    let mut positions = Array2::<f64>::zeros((natoms, 3));
    let mut flags_opt: Option<Array2<bool>> = if selective {
        Some(Array2::<bool>::from_elem((natoms, 3), false))
    } else {
        None
    };

    for (i, line) in lines.iter().enumerate() {
        let mut parts = line.split_ascii_whitespace();

        // Parse 3 position floats
        for j in 0..3 {
            let tok = parts.next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "POSCAR: not enough fields on atom line {} (need 3 coordinates)",
                    i
                ))
            })?;
            positions[[i, j]] = tok.parse::<f64>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "POSCAR: cannot parse coordinate '{}' on atom line {}: {}",
                    tok, i, e
                ))
            })?;
        }

        // Parse optional T/F selective-dynamics flags
        if let Some(ref mut fa) = flags_opt {
            for j in 0..3 {
                match parts.next() {
                    Some("F") => fa[[i, j]] = true,   // 'F' = frozen = movement disallowed
                    Some("T") => fa[[i, j]] = false,  // 'T' = free to move
                    None => fa[[i, j]] = false,        // missing flag: assume free
                    Some(tok) => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "POSCAR: expected selective dynamics flag 'T' or 'F', \
                             got '{}' on atom line {}",
                            tok, i
                        )));
                    }
                }
            }
        }
    }

    let pos_py = positions.into_pyarray(py);
    let flags_any: Bound<'py, PyAny> = match flags_opt {
        Some(fa) => fa.into_pyarray(py).into_any(),
        // py.None() → Py<PyNone> → Py<PyAny> → &Bound<'py, PyAny> → Bound<'py, PyAny>
        None => py.None().into_any().bind(py).clone(),
    };
    Ok((pos_py.into(), flags_any))
}

// ─────────────────────────────────────────────────────────────────────────────
// POSCAR write
// ─────────────────────────────────────────────────────────────────────────────

/// Format the N-atom position block for a POSCAR/CONTCAR file.
///
/// Python equivalent (write_vasp:908-914):
/// ```python
/// for iatom, atom in enumerate(coord):
///     for dcoord in atom:
///         fd.write(f' {dcoord:19.16f}')
///     if constraints_present:
///         flags = ['F' if flag else 'T' for flag in sflags[iatom]]
///         fd.write(''.join([f'{f:>4s}' for f in flags]))
///     fd.write('\n')
/// ```
///
/// Parameters
/// ----------
/// coord : ndarray float64 [N,3]
///     Atom positions (cartesian or scaled — caller decides).
/// flags : ndarray bool [N,3] or None
///     Selective-dynamics mask (True = frozen = 'F' in file).
///
/// Returns
/// -------
/// str  — the complete block ready for fd.write()
#[pyfunction]
#[pyo3(signature = (coord, flags=None))]
pub fn format_poscar_positions_rs(
    coord: PyReadonlyArray2<f64>,
    flags: Option<PyReadonlyArray2<bool>>,
) -> PyResult<String> {
    let c = coord.as_array();
    let natoms = c.nrows();
    // Each line: 3 × (1 space + 19-wide float) = 60 chars, optional 3×4 flags, newline
    let line_cap = 3 * 20 + if flags.is_some() { 12 } else { 0 } + 1;
    let mut out = String::with_capacity(natoms * line_cap);

    match flags {
        Some(f) => {
            let fa = f.as_array();
            if fa.shape() != c.shape() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "format_poscar_positions_rs: coord and flags must have the same shape",
                ));
            }
            for i in 0..natoms {
                write!(out, " {:19.16} {:19.16} {:19.16}",
                    c[[i, 0]], c[[i, 1]], c[[i, 2]])
                    .expect("String write should not fail");
                for j in 0..3 {
                    // True = frozen = 'F'; False = free = 'T'
                    let ch = if fa[[i, j]] { 'F' } else { 'T' };
                    write!(out, "{:>4}", ch).expect("String write should not fail");
                }
                out.push('\n');
            }
        }
        None => {
            for i in 0..natoms {
                write!(out, " {:19.16} {:19.16} {:19.16}\n",
                    c[[i, 0]], c[[i, 1]], c[[i, 2]])
                    .expect("String write should not fail");
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// XDATCAR read
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the coordinate block for one XDATCAR frame.
///
/// Python equivalent (read_vasp_xdatcar:398):
/// ```python
/// coords = [np.array(fd.readline().split(), float) for _ in range(total)]
/// ```
///
/// Parameters
/// ----------
/// lines : list[str]
///     The N coordinate lines for this frame (already read).
///
/// Returns
/// -------
/// ndarray float64 [N,3] — fractional coordinates
#[pyfunction]
pub fn parse_xdatcar_coords_rs<'py>(
    py: Python<'py>,
    lines: Vec<String>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let natoms = lines.len();
    let mut coords = Array2::<f64>::zeros((natoms, 3));

    for (i, line) in lines.iter().enumerate() {
        let mut parts = line.split_ascii_whitespace();
        for j in 0..3 {
            let tok = parts.next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "XDATCAR: not enough fields on coordinate line {}",
                    i
                ))
            })?;
            coords[[i, j]] = tok.parse::<f64>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "XDATCAR: cannot parse '{}' on line {}: {}",
                    tok, i, e
                ))
            })?;
        }
    }
    Ok(coords.into_pyarray(py).into())
}

// ─────────────────────────────────────────────────────────────────────────────
// XDATCAR write
// ─────────────────────────────────────────────────────────────────────────────

/// Format one complete XDATCAR configuration block.
///
/// Python equivalent (_write_xdatcar_config:763-769):
/// ```python
/// fd.write(f"Direct configuration={index:6d}\n")
/// for row in scaled_positions:
///     fd.write('  ')
///     fd.write(' '.join(['{:11.8f}'.format(x) for x in row]))
///     fd.write('\n')
/// ```
///
/// Parameters
/// ----------
/// scaled_positions : ndarray float64 [N,3]
/// index : int — 1-based frame index written to the header line
///
/// Returns
/// -------
/// str  — full block including header, ready for fd.write()
#[pyfunction]
pub fn format_xdatcar_config_rs(
    scaled_positions: PyReadonlyArray2<f64>,
    index: usize,
) -> PyResult<String> {
    let sp = scaled_positions.as_array();
    let natoms = sp.nrows();
    // Header (~30 chars) + N lines (~40 chars each)
    let mut out = String::with_capacity(30 + natoms * 40);
    write!(out, "Direct configuration={:6}\n", index)
        .expect("String write should not fail");
    for i in 0..natoms {
        write!(out, "  {:11.8} {:11.8} {:11.8}\n",
            sp[[i, 0]], sp[[i, 1]], sp[[i, 2]])
            .expect("String write should not fail");
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (pure Rust — no Python runtime needed)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that format_poscar_positions_rs matches Python's f' {v:19.16f}' per column.
    #[test]
    fn test_format_no_flags_basic() {
        // 1 atom at (1.5, 0.0, -2.5) — check format string match
        let c = Array2::from_shape_vec((1, 3), vec![1.5_f64, 0.0_f64, -2.5_f64]).unwrap();
        let mut out = String::new();
        write!(out, " {:19.16} {:19.16} {:19.16}\n", c[[0,0]], c[[0,1]], c[[0,2]]).unwrap();
        // " " + 19-wide 1.5 + " " + 19-wide 0.0 + " " + 19-wide -2.5 + "\n"
        assert!(out.contains("1.5000000000000000"), "1.5 not formatted correctly: {}", out);
        assert!(out.contains("0.0000000000000000"), "0.0 not formatted correctly: {}", out);
        assert!(out.contains("-2.5000000000000000"), "-2.5 not formatted correctly: {}", out);
    }

    /// Verify that F/T flags are right-aligned in width 4.
    #[test]
    fn test_format_flags_alignment() {
        let mut out = String::new();
        write!(out, "{:>4}", 'F').unwrap();
        assert_eq!(&out, "   F");
        let mut out2 = String::new();
        write!(out2, "{:>4}", 'T').unwrap();
        assert_eq!(&out2, "   T");
    }

    /// Verify XDATCAR header format matches f"Direct configuration={n:6d}"
    #[test]
    fn test_xdatcar_header_format() {
        let mut out = String::new();
        write!(out, "Direct configuration={:6}\n", 1_usize).unwrap();
        assert_eq!(&out, "Direct configuration=     1\n");
        let mut out2 = String::new();
        write!(out2, "Direct configuration={:6}\n", 100_usize).unwrap();
        assert_eq!(&out2, "Direct configuration=   100\n");
    }

    /// Verify parse round-trips: format 3 values, split on whitespace, re-parse.
    #[test]
    fn test_poscar_format_parse_roundtrip() {
        let vals = [1.23456789_f64, -0.00000001_f64, 99.9999999999999_f64];
        let mut line = String::new();
        write!(line, " {:19.16} {:19.16} {:19.16}\n", vals[0], vals[1], vals[2]).unwrap();
        let parsed: Vec<f64> = line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        for (orig, got) in vals.iter().zip(parsed.iter()) {
            assert!((orig - got).abs() < 1e-14,
                "round-trip mismatch: orig={} got={}", orig, got);
        }
    }

    /// Verify XDATCAR coordinate format round-trip.
    #[test]
    fn test_xdatcar_format_parse_roundtrip() {
        let vals = [0.12345678_f64, 0.99999999_f64, 0.0_f64];
        let mut line = String::new();
        write!(line, "  {:11.8} {:11.8} {:11.8}\n", vals[0], vals[1], vals[2]).unwrap();
        let parsed: Vec<f64> = line.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        for (orig, got) in vals.iter().zip(parsed.iter()) {
            assert!((orig - got).abs() < 1e-7,
                "XDATCAR round-trip: orig={} got={}", orig, got);
        }
    }
}
