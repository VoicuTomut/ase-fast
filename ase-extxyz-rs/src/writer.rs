//! Fast atom-block formatter for extxyz write path.
//!
//! Replaces the Python loop:
//!     for i in range(natoms):
//!         fileobj.write(fmt % tuple(data[i]))
//!
//! with a single Rust pass over contiguous memory.
//!
//! Column format rules (fixed per type, matching output_column_format fmt_map):
//!   's' (string/symbol)  → left-justify to sym_width, no leading separator
//!   'f' (float64)        → right-justify %16.8f  (%>16.8 in Rust)
//!   'i' (int64)          → right-justify %8d
//!   'b' (bool, as u8)    → ' %.1s' → writes "  T" or "  F" (2 spaces + char)
//!
//! The separator between columns comes from Python's ' '.join(formats):
//!   - non-bool column: one space written before the value
//!   - bool column: two spaces (join separator + the ' ' in ' %.1s')

use std::fmt::Write as FmtWrite;
use ndarray::ArrayView2;

/// Decode one row of UTF-32 LE codepoints (zero-terminated) into a String.
#[inline(always)]
fn decode_symbol(codepoints: &[u32]) -> String {
    let mut s = String::with_capacity(codepoints.len());
    for &cp in codepoints {
        if cp == 0 { break; }
        if let Some(ch) = char::from_u32(cp) {
            s.push(ch);
        }
    }
    s
}

/// Fast writer: symbols are passed as a (natoms, max_chars) uint32 UTF-32 array.
/// Avoids creating Python string objects — the entire symbol column is processed
/// in Rust from the raw numpy buffer.  Zero-copy from the Python side.
pub fn format_atom_block_codepoints(
    sym_codepoints: ArrayView2<u32>,   // (natoms, max_str_len), UTF-32 LE
    sym_width: usize,
    float_mat: ArrayView2<f64>,
    int_mat: ArrayView2<i64>,
    bool_mat: ArrayView2<u8>,
    col_types: &[u8],
) -> String {
    let natoms = sym_codepoints.nrows();
    let max_len = sym_codepoints.ncols();
    let n_float_cols = float_mat.ncols();
    let n_int_cols = int_mat.ncols();
    let n_bool_cols = bool_mat.ncols();

    let line_est = sym_width + n_float_cols * 17 + n_int_cols * 9 + n_bool_cols * 3 + 2;
    let mut buf = String::with_capacity(natoms * line_est);

    for i in 0..natoms {
        let mut float_col = 0usize;
        let mut int_col   = 0usize;
        let mut bool_col  = 0usize;

        for &ct in col_types {
            match ct {
                b's' => {
                    // Decode symbol from UTF-32 codepoints (no Python objects)
                    let row = sym_codepoints.row(i);
                    let sym = decode_symbol(row.as_slice().unwrap_or(&[]));
                    write!(buf, "{:<sw$}", sym, sw = sym_width).unwrap();
                }
                b'f' => {
                    buf.push(' ');
                    write!(buf, "{:16.8}", float_mat[[i, float_col]]).unwrap();
                    float_col += 1;
                }
                b'i' => {
                    buf.push(' ');
                    write!(buf, "{:8}", int_mat[[i, int_col]]).unwrap();
                    int_col += 1;
                }
                b'b' => {
                    buf.push(' ');
                    buf.push(' ');
                    let ch = if bool_mat[[i, bool_col]] != 0 { 'T' } else { 'F' };
                    buf.push(ch);
                    bool_col += 1;
                }
                _ => {}
            }
        }
        buf.push('\n');
    }

    buf
}

/// Build the complete atom block (all `natoms` lines) as a single String.
///
/// # Arguments
/// * `symbols`   – chemical symbols, length = natoms
/// * `sym_width` – left-justify field width for symbols (from `%-Ns` format)
/// * `float_mat` – shape (natoms, n_float_cols), f64, row-major
/// * `int_mat`   – shape (natoms, n_int_cols),   i64, row-major
/// * `bool_mat`  – shape (natoms, n_bool_cols),  u8 (0=F, nonzero=T), row-major
/// * `col_types` – column order: b's'=symbol, b'f'=float, b'i'=int, b'b'=bool
pub fn format_atom_block(
    symbols: &[String],
    sym_width: usize,
    float_mat: ArrayView2<f64>,
    int_mat: ArrayView2<i64>,
    bool_mat: ArrayView2<u8>,
    col_types: &[u8],
) -> String {
    let natoms = symbols.len();
    let n_float_cols = float_mat.ncols();
    let n_int_cols = int_mat.ncols();
    let n_bool_cols = bool_mat.ncols();

    // Estimate per-line capacity to avoid repeated allocations.
    // sym_width + n_float*(1+16) + n_int*(1+8) + n_bool*(1+2) + newline
    let line_est = sym_width
        + n_float_cols * 17
        + n_int_cols * 9
        + n_bool_cols * 3
        + 2;
    let mut buf = String::with_capacity(natoms * line_est);

    for i in 0..natoms {
        let mut float_col = 0usize;
        let mut int_col   = 0usize;
        let mut bool_col  = 0usize;
        let mut is_first  = true;

        for &ct in col_types {
            match ct {
                b's' => {
                    // Symbol column: always first, no leading separator.
                    // Left-justify to sym_width with space padding.
                    write!(buf, "{:<sw$}", symbols[i], sw = sym_width).unwrap();
                    is_first = false;
                }
                b'f' => {
                    if !is_first { buf.push(' '); }
                    // %16.8f : right-justified, 16 wide, 8 decimal places.
                    // Rust Display with precision uses fixed-point notation.
                    write!(buf, "{:16.8}", float_mat[[i, float_col]]).unwrap();
                    float_col += 1;
                    is_first = false;
                }
                b'i' => {
                    if !is_first { buf.push(' '); }
                    // %8d : right-justified, 8 wide.
                    write!(buf, "{:8}", int_mat[[i, int_col]]).unwrap();
                    int_col += 1;
                    is_first = false;
                }
                b'b' => {
                    // ' %.1s' format: 1 char ('T'/'F') with a leading literal space.
                    // Plus the join separator space = 2 spaces total before the char.
                    if !is_first { buf.push(' '); }  // join separator
                    buf.push(' ');                    // leading space from ' %.1s'
                    let ch = if bool_mat[[i, bool_col]] != 0 { 'T' } else { 'F' };
                    buf.push(ch);
                    bool_col += 1;
                    is_first = false;
                }
                _ => {} // unknown col type: skip (shouldn't happen)
            }
        }
        buf.push('\n');
    }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn empty_f64(natoms: usize) -> Array2<f64> { Array2::zeros((natoms, 0)) }
    fn empty_i64(natoms: usize) -> Array2<i64> { Array2::zeros((natoms, 0)) }
    fn empty_u8 (natoms: usize) -> Array2<u8>  { Array2::zeros((natoms, 0)) }

    #[test]
    fn test_single_atom_float() {
        let syms = vec!["Cu".to_string()];
        let mut pos = Array2::<f64>::zeros((1, 3));
        pos[[0, 0]] = 0.0;
        pos[[0, 1]] = 1.80750000;
        pos[[0, 2]] = 1.80750000;
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(1).view(), empty_u8(1).view(),
            b"sfff",
        );
        // Python: '%-2s %16.8f %16.8f %16.8f\n' % ('Cu', 0.0, 1.8075, 1.8075)
        let expected = "Cu       0.00000000       1.80750000       1.80750000\n";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_h_symbol_padded() {
        let syms = vec!["H".to_string()];
        let pos = Array2::<f64>::zeros((1, 3));
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(1).view(), empty_u8(1).view(),
            b"sfff",
        );
        // 'H ' left-justified to width 2
        assert!(result.starts_with("H "));
    }

    #[test]
    fn test_negative_position() {
        let syms = vec!["O".to_string()];
        let mut pos = Array2::<f64>::zeros((1, 3));
        pos[[0, 0]] = -1.5;
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(1).view(), empty_u8(1).view(),
            b"sfff",
        );
        // '%16.8f' % -1.5 = '      -1.50000000'  (16 chars)
        assert!(result.contains("      -1.50000000"));
    }

    #[test]
    fn test_int_col() {
        let syms = vec!["Fe".to_string()];
        let pos = Array2::<f64>::zeros((1, 3));
        let mut charges = Array2::<i64>::zeros((1, 1));
        charges[[0, 0]] = 2;
        let result = format_atom_block(
            &syms, 2,
            pos.view(), charges.view(), empty_u8(1).view(),
            b"sfffi",
        );
        assert!(result.contains("       2"));
    }

    #[test]
    fn test_bool_col() {
        let syms = vec!["C".to_string()];
        let pos = Array2::<f64>::zeros((1, 3));
        let mut mask = Array2::<u8>::zeros((1, 1));
        mask[[0, 0]] = 1; // True
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(1).view(), mask.view(),
            b"sfffb",
        );
        assert!(result.contains("  T"));
    }

    #[test]
    fn test_bool_false() {
        let syms = vec!["C".to_string()];
        let pos = Array2::<f64>::zeros((1, 3));
        let mask = Array2::<u8>::zeros((1, 1)); // all False
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(1).view(), mask.view(),
            b"sfffb",
        );
        assert!(result.contains("  F"));
    }

    #[test]
    fn test_multiline() {
        let syms = vec!["Cu".to_string(), "Cu".to_string()];
        let mut pos = Array2::<f64>::zeros((2, 3));
        pos[[1, 1]] = 1.80750000;
        pos[[1, 2]] = 1.80750000;
        let result = format_atom_block(
            &syms, 2,
            pos.view(), empty_i64(2).view(), empty_u8(2).view(),
            b"sfff",
        );
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("Cu"));
        assert!(lines[1].starts_with("Cu"));
    }
}
