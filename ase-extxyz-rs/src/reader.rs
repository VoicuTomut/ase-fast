//! Fast atom-line parser for extxyz read path.
//!
//! Replaces the Python loop:
//!     data = []
//!     for _ in range(natoms):
//!         line = next(lines)
//!         vals = line.split()
//!         row = tuple(conv(val) for conv, val in zip(convs, vals))
//!         data.append(row)
//!     data = np.array(data, dtype)
//!
//! col_types encodes the converter per token:
//!   b'f' → parse as f64
//!   b'i' → parse as i64
//!   b's' → keep as String
//!   b'b' → parse as bool ('T'/'True' → 1, 'F'/'False' → 0)

/// Result of parsing N atom lines.
#[derive(Debug)]
pub struct ParsedAtomData {
    /// Shape (natoms, n_float_cols), row-major
    pub floats: Vec<f64>,
    pub n_float_cols: usize,
    /// Shape (natoms, n_int_cols), row-major
    pub ints: Vec<i64>,
    pub n_int_cols: usize,
    /// Shape (natoms, n_bool_cols) as 0/1, row-major
    pub bools: Vec<u8>,
    pub n_bool_cols: usize,
    /// One Vec<String> per string column, each of length natoms
    pub strs: Vec<Vec<String>>,
}

/// Parse `lines` where each line has tokens in the order given by `col_types`.
///
/// Returns `Err(msg)` on the first line that cannot be parsed.
pub fn parse_atom_lines(lines: &[String], col_types: &[u8]) -> Result<ParsedAtomData, String> {
    let natoms = lines.len();
    let n_float_cols = col_types.iter().filter(|&&c| c == b'f').count();
    let n_int_cols   = col_types.iter().filter(|&&c| c == b'i').count();
    let n_bool_cols  = col_types.iter().filter(|&&c| c == b'b').count();
    let n_str_cols   = col_types.iter().filter(|&&c| c == b's').count();

    let mut floats = vec![0.0_f64; natoms * n_float_cols];
    let mut ints   = vec![0_i64;  natoms * n_int_cols];
    let mut bools  = vec![0_u8;   natoms * n_bool_cols];
    let mut strs: Vec<Vec<String>> = (0..n_str_cols).map(|_| Vec::with_capacity(natoms)).collect();

    for (row, line) in lines.iter().enumerate() {
        let mut tokens = line.split_ascii_whitespace();
        let mut float_col = 0usize;
        let mut int_col   = 0usize;
        let mut bool_col  = 0usize;
        let mut str_col   = 0usize;

        for &ct in col_types {
            let tok = tokens.next().ok_or_else(|| {
                format!("row {}: expected {} tokens, ran out", row, col_types.len())
            })?;

            match ct {
                b'f' => {
                    let v = tok.parse::<f64>().map_err(|e| {
                        format!("row {}: cannot parse {:?} as float: {}", row, tok, e)
                    })?;
                    floats[row * n_float_cols + float_col] = v;
                    float_col += 1;
                }
                b'i' => {
                    let v = tok.parse::<i64>().map_err(|e| {
                        format!("row {}: cannot parse {:?} as int: {}", row, tok, e)
                    })?;
                    ints[row * n_int_cols + int_col] = v;
                    int_col += 1;
                }
                b'b' => {
                    let v: u8 = match tok {
                        "T" | "True" | "TRUE" | "true"   => 1,
                        "F" | "False" | "FALSE" | "false" => 0,
                        other => return Err(format!(
                            "row {}: cannot parse {:?} as bool", row, other
                        )),
                    };
                    bools[row * n_bool_cols + bool_col] = v;
                    bool_col += 1;
                }
                b's' => {
                    strs[str_col].push(tok.to_string());
                    str_col += 1;
                }
                _ => {} // unknown: skip token
            }
        }
    }

    Ok(ParsedAtomData {
        floats,
        n_float_cols,
        ints,
        n_int_cols,
        bools,
        n_bool_cols,
        strs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_species_pos() {
        let lines = vec![
            "Cu       0.00000000       0.00000000       0.00000000".to_string(),
            "Cu       0.00000000       1.80750000       1.80750000".to_string(),
        ];
        let col_types = b"sfff";
        let result = parse_atom_lines(&lines, col_types).unwrap();
        assert_eq!(result.n_float_cols, 3);
        assert_eq!(result.strs.len(), 1);
        assert_eq!(result.floats.len(), 6);
        assert!((result.floats[3] - 0.0).abs() < 1e-10);
        assert!((result.floats[4] - 1.8075).abs() < 1e-6);
        assert_eq!(result.strs[0], vec!["Cu", "Cu"]);
    }

    #[test]
    fn test_bool_col() {
        let lines = vec![
            "Cu  0.0  0.0  0.0  T".to_string(),
            "H   0.0  0.0  1.0  F".to_string(),
        ];
        let result = parse_atom_lines(&lines, b"sfffb").unwrap();
        assert_eq!(result.n_bool_cols, 1);
        assert_eq!(result.bools[0], 1); // T
        assert_eq!(result.bools[1], 0); // F
    }

    #[test]
    fn test_int_col() {
        let lines = vec!["Fe  0.0  0.0  0.0  2".to_string()];
        let result = parse_atom_lines(&lines, b"sfffi").unwrap();
        assert_eq!(result.n_int_cols, 1);
        assert_eq!(result.ints[0], 2);
    }

    #[test]
    fn test_bad_float() {
        let lines = vec!["Cu  not_a_float  0.0  0.0".to_string()];
        let err = parse_atom_lines(&lines, b"sfff").unwrap_err();
        assert!(err.contains("cannot parse"));
    }

    #[test]
    fn test_too_few_tokens() {
        let lines = vec!["Cu  0.0".to_string()]; // missing two floats
        let err = parse_atom_lines(&lines, b"sfff").unwrap_err();
        assert!(err.contains("ran out"));
    }
}
