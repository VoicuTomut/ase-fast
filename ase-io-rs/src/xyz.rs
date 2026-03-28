//! Simple XYZ IO kernels — Rust ports of the per-atom Python loops in
//! ase/io/xyz.py.
//!
//! Two functions are exposed:
//!   parse_xyz_block_rs   — read N atom lines → (symbols, positions)
//!   format_xyz_block_rs  — write N atoms → block string

use std::fmt::Write as FmtWrite;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Simple XYZ read
// ─────────────────────────────────────────────────────────────────────────────

/// Parse N atom lines from a simple XYZ file.
///
/// Python equivalent (read_xyz:26-31):
/// ```python
/// for _ in range(natoms):
///     line = lines.pop(0)
///     symbol, x, y, z = line.split()[:4]
///     symbol = symbol.lower().capitalize()
///     symbols.append(symbol)
///     positions.append([float(x), float(y), float(z)])
/// ```
///
/// Parameters
/// ----------
/// lines : list[str]
///     The N atom lines (already extracted from the full file).
///
/// Returns
/// -------
/// (symbols: list[str], positions: ndarray float64 [N,3])
#[pyfunction]
pub fn parse_xyz_block_rs<'py>(
    py: Python<'py>,
    lines: Vec<String>,
) -> PyResult<(Vec<String>, Bound<'py, PyArray2<f64>>)> {
    let natoms = lines.len();
    let mut symbols: Vec<String> = Vec::with_capacity(natoms);
    let mut positions = Array2::<f64>::zeros((natoms, 3));

    for (i, line) in lines.iter().enumerate() {
        let mut parts = line.split_ascii_whitespace();

        // Symbol — first token, capitalized like Python's .lower().capitalize()
        let sym_raw = parts.next().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "XYZ: empty line at atom index {}",
                i
            ))
        })?;
        let sym_lower = sym_raw.to_lowercase();
        let sym_cap = capitalize_first(&sym_lower);
        symbols.push(sym_cap);

        // 3 position floats
        for j in 0..3 {
            let tok = parts.next().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "XYZ: not enough position fields on atom line {}",
                    i
                ))
            })?;
            positions[[i, j]] = tok.parse::<f64>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "XYZ: cannot parse coordinate '{}' on line {}: {}",
                    tok, i, e
                ))
            })?;
        }
    }

    Ok((symbols, positions.into_pyarray(py).into()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple XYZ write
// ─────────────────────────────────────────────────────────────────────────────

/// Format N atoms as simple XYZ atom lines.
///
/// Python equivalent (write_xyz:42-43, fmt='%22.15f'):
/// ```python
/// for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
///     fileobj.write('%-2s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z))
/// ```
///
/// Parameters
/// ----------
/// symbols   : list[str]  — chemical symbols (length N)
/// positions : ndarray float64 [N,3]
///
/// Returns
/// -------
/// str  — the block (without the natoms / comment header lines)
#[pyfunction]
pub fn format_xyz_block_rs(
    symbols: Vec<String>,
    positions: PyReadonlyArray2<f64>,
) -> PyResult<String> {
    let p = positions.as_array();
    let natoms = symbols.len();
    if p.nrows() != natoms {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "format_xyz_block_rs: symbols length ({}) != positions rows ({})",
            natoms,
            p.nrows()
        )));
    }
    // Each line: "%-2s %22.15f %22.15f %22.15f\n" ≈ 75 chars
    let mut out = String::with_capacity(natoms * 75);
    for i in 0..natoms {
        write!(out, "{:<2} {:22.15} {:22.15} {:22.15}\n",
            &symbols[i], p[[i, 0]], p[[i, 1]], p[[i, 2]])
            .expect("String write should not fail");
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Equivalent to Python's str.lower().capitalize():
/// uppercase the first char, lowercase the rest.
#[inline]
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => {
            let upper: String = c.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalize_first() {
        assert_eq!(capitalize_first("si"), "Si");
        assert_eq!(capitalize_first("fe"), "Fe");
        assert_eq!(capitalize_first("h"), "H");
        assert_eq!(capitalize_first("ge"), "Ge");
        // Python: "SI".lower() = "si", "si".capitalize() = "Si"
        assert_eq!(capitalize_first("sI".to_lowercase().as_str()), "Si");
        // Python: "H2O" — this function won't be called with multi-char non-alpha
        // but should still not panic
        assert_eq!(capitalize_first("h2"), "H2");
    }

    #[test]
    fn test_xyz_format_roundtrip() {
        let vals = [1.23456789012345_f64, -0.5_f64, 99.9_f64];
        let mut line = String::new();
        write!(line, "{:<2} {:22.15} {:22.15} {:22.15}\n",
            "Si", vals[0], vals[1], vals[2]).unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts[0], "Si");
        let x: f64 = parts[1].parse().unwrap();
        let y: f64 = parts[2].parse().unwrap();
        let z: f64 = parts[3].parse().unwrap();
        assert!((x - vals[0]).abs() < 1e-13);
        assert!((y - vals[1]).abs() < 1e-13);
        assert!((z - vals[2]).abs() < 1e-13);
    }

    #[test]
    fn test_xyz_symbol_left_aligned() {
        // "%-2s" leaves single-char symbols left-padded to 2: "H " (space on right)
        let mut out = String::new();
        write!(out, "{:<2}", "H").unwrap();
        assert_eq!(&out, "H "); // left-aligned, 2 wide
        let mut out2 = String::new();
        write!(out2, "{:<2}", "Fe").unwrap();
        assert_eq!(&out2, "Fe"); // exactly 2, no padding needed
    }
}
