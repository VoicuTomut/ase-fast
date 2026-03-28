//! Cell-list neighbor search — exact port of ASE's `primitive_neighbor_list`
//! Python/NumPy implementation.
//!
//! All index conventions, bin formulas, and edge cases mirror the Python source
//! so that the contract tests (TestNeighborlistContracts) pass bit-for-bit.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

// ── Public entry point ──────────────────────────────────────────────────────

/// Build the full neighbor list.  Returns (i, j, d, D, S).
pub fn build_neighbor_list(
    pbc: &[bool; 3],
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    cutoff: f64,
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> Result<(Array1<i64>, Array1<i64>, Array1<f64>, Array2<f64>, Array2<i64>), String> {
    let n_atoms = positions.nrows();

    // Handle empty input.
    if n_atoms == 0 {
        return Ok((
            Array1::zeros(0),
            Array1::zeros(0),
            Array1::zeros(0),
            Array2::zeros((0, 3)),
            Array2::zeros((0, 3)),
        ));
    }

    // ── 1. Cell geometry ─────────────────────────────────────────────────

    // Cross products to get face normals (reciprocal-space direction).
    let a = [cell[[0,0]], cell[[0,1]], cell[[0,2]]];
    let b = [cell[[1,0]], cell[[1,1]], cell[[1,2]]];
    let c = [cell[[2,0]], cell[[2,1]], cell[[2,2]]];

    let b_cross_c = cross(&b, &c);
    let c_cross_a = cross(&c, &a);
    let a_cross_b = cross(&a, &b);

    let vol = dot(&a, &b_cross_c).abs();

    // face_dist_c[k] = distance between the two faces perpendicular to axis k.
    let face_dist = [
        vol / norm(&b_cross_c),
        vol / norm(&c_cross_a),
        vol / norm(&a_cross_b),
    ];

    // ── 2. Bin count ──────────────────────────────────────────────────────

    let bin_size = cutoff.max(3.0); // minimum 3 Å bin size (same as Python)

    let mut nbins = [
        ((face_dist[0] / bin_size) as usize).max(1),
        ((face_dist[1] / bin_size) as usize).max(1),
        ((face_dist[2] / bin_size) as usize).max(1),
    ];

    // Cap total bins at max_nbins.
    while nbins[0] * nbins[1] * nbins[2] > max_nbins {
        nbins[0] = (nbins[0] / 2).max(1);
        nbins[1] = (nbins[1] / 2).max(1);
        nbins[2] = (nbins[2] / 2).max(1);
    }

    // Neighbour shell search extent (same formula as Python).
    let neigh_x = ceil_div_usize(bin_size * nbins[0] as f64, face_dist[0]);
    let neigh_y = ceil_div_usize(bin_size * nbins[1] as f64, face_dist[1]);
    let neigh_z = ceil_div_usize(bin_size * nbins[2] as f64, face_dist[2]);

    // For non-periodic, single-bin axes the search extent is 0.
    let neigh = [
        if nbins[0] == 1 && !pbc[0] { 0 } else { neigh_x },
        if nbins[1] == 1 && !pbc[1] { 0 } else { neigh_y },
        if nbins[2] == 1 && !pbc[2] { 0 } else { neigh_z },
    ];

    // ── 3. Convert positions to scaled coords ────────────────────────────

    // We need the inverse of cell (cell rows are lattice vectors).
    let cart_positions: Array2<f64>;
    let scaled: Array2<f64>;

    if use_scaled_positions {
        scaled = positions.to_owned();
        cart_positions = mat_mul_rows(&scaled, cell);
    } else {
        cart_positions = positions.to_owned();
        scaled = solve3x3(cell, &cart_positions)?;
    }

    // ── 4. Assign atoms to bins ───────────────────────────────────────────

    let mut bin_idx = vec![[0i64; 3]; n_atoms];
    let mut cell_shift = vec![[0i64; 3]; n_atoms]; // PBC wrap shift

    for i in 0..n_atoms {
        for c in 0..3 {
            let raw = (scaled[[i, c]] * nbins[c] as f64).floor() as i64;
            if pbc[c] {
                let (shift, idx) = divmod_i64(raw, nbins[c] as i64);
                cell_shift[i][c] = shift;
                bin_idx[i][c] = idx;
            } else {
                bin_idx[i][c] = raw.clamp(0, nbins[c] as i64 - 1);
            }
        }
    }

    // Linearise bin index.
    let linear_bin: Vec<usize> = (0..n_atoms)
        .map(|i| {
            (bin_idx[i][0] as usize)
                + nbins[0] * (bin_idx[i][1] as usize + nbins[1] * bin_idx[i][2] as usize)
        })
        .collect();

    // Sort atoms by bin for cache-friendly iteration.
    let mut atom_order: Vec<usize> = (0..n_atoms).collect();
    atom_order.sort_by_key(|&i| linear_bin[i]);

    // Build bin contents: bin_atoms[b] = list of atom indices in bin b.
    let total_bins = nbins[0] * nbins[1] * nbins[2];
    let mut bin_atoms: Vec<Vec<usize>> = vec![Vec::new(); total_bins];
    for &a in &atom_order {
        bin_atoms[linear_bin[a]].push(a);
    }

    // ── 5. Neighbor search: iterate over all bins and their 27 shells ────

    let cutoff2 = cutoff * cutoff;

    // Accumulators.
    let mut out_i: Vec<i64> = Vec::new();
    let mut out_j: Vec<i64> = Vec::new();
    let mut out_d: Vec<f64> = Vec::new();
    let mut out_d_vec: Vec<[f64; 3]> = Vec::new();
    let mut out_s: Vec<[i64; 3]> = Vec::new();

    // Iterate over every bin (bx, by, bz).
    for bz in 0..nbins[2] as i64 {
        for by in 0..nbins[1] as i64 {
            for bx in 0..nbins[0] as i64 {
                let bin_b = (bx as usize)
                    + nbins[0] * (by as usize + nbins[1] * bz as usize);
                let atoms_in_b = &bin_atoms[bin_b];
                if atoms_in_b.is_empty() {
                    continue;
                }

                // Iterate over the 27-shell of neighbour bins.
                for dz in -(neigh[2] as i64)..=(neigh[2] as i64) {
                    for dy in -(neigh[1] as i64)..=(neigh[1] as i64) {
                        for dx in -(neigh[0] as i64)..=(neigh[0] as i64) {
                            let (shift_x, nbx) = divmod_i64(bx + dx, nbins[0] as i64);
                            let (shift_y, nby) = divmod_i64(by + dy, nbins[1] as i64);
                            let (shift_z, nbz) = divmod_i64(bz + dz, nbins[2] as i64);

                            let nbin = (nbx as usize)
                                + nbins[0] * (nby as usize + nbins[1] * nbz as usize);
                            let atoms_in_nb = &bin_atoms[nbin];

                            for &atom_i in atoms_in_b {
                                for &atom_j in atoms_in_nb {
                                    // Net cell shift: shift from bin wrap + atom cell
                                    // shift difference (same as Python cell_shift_ic).
                                    let s = [
                                        shift_x + cell_shift[atom_i][0]
                                            - cell_shift[atom_j][0],
                                        shift_y + cell_shift[atom_i][1]
                                            - cell_shift[atom_j][1],
                                        shift_z + cell_shift[atom_i][2]
                                            - cell_shift[atom_j][2],
                                    ];

                                    // Skip pairs crossing non-periodic boundaries.
                                    if (!pbc[0] && s[0] != 0)
                                        || (!pbc[1] && s[1] != 0)
                                        || (!pbc[2] && s[2] != 0)
                                    {
                                        continue;
                                    }

                                    // Skip self-pairs that don't cross a boundary.
                                    if !self_interaction
                                        && atom_i == atom_j
                                        && s[0] == 0
                                        && s[1] == 0
                                        && s[2] == 0
                                    {
                                        continue;
                                    }

                                    // D = pos[j] - pos[i] + S @ cell
                                    let s_dot_cell = [
                                        s[0] as f64 * cell[[0,0]]
                                            + s[1] as f64 * cell[[1,0]]
                                            + s[2] as f64 * cell[[2,0]],
                                        s[0] as f64 * cell[[0,1]]
                                            + s[1] as f64 * cell[[1,1]]
                                            + s[2] as f64 * cell[[2,1]],
                                        s[0] as f64 * cell[[0,2]]
                                            + s[1] as f64 * cell[[1,2]]
                                            + s[2] as f64 * cell[[2,2]],
                                    ];

                                    let d_vec = [
                                        cart_positions[[atom_j, 0]]
                                            - cart_positions[[atom_i, 0]]
                                            + s_dot_cell[0],
                                        cart_positions[[atom_j, 1]]
                                            - cart_positions[[atom_i, 1]]
                                            + s_dot_cell[1],
                                        cart_positions[[atom_j, 2]]
                                            - cart_positions[[atom_i, 2]]
                                            + s_dot_cell[2],
                                    ];

                                    let d2 = d_vec[0] * d_vec[0]
                                        + d_vec[1] * d_vec[1]
                                        + d_vec[2] * d_vec[2];

                                    if d2 < cutoff2 {
                                        out_i.push(atom_i as i64);
                                        out_j.push(atom_j as i64);
                                        out_d.push(d2.sqrt());
                                        out_d_vec.push(d_vec);
                                        out_s.push(s);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── 6. Sort by i (ascending) ─────────────────────────────────────────

    let mut order: Vec<usize> = (0..out_i.len()).collect();
    order.sort_by_key(|&k| out_i[k]);

    // ── 7. Pack into ndarray outputs ──────────────────────────────────────

    let n_pairs = out_i.len();
    let mut arr_i = Array1::<i64>::zeros(n_pairs);
    let mut arr_j = Array1::<i64>::zeros(n_pairs);
    let mut arr_d = Array1::<f64>::zeros(n_pairs);
    let mut arr_d_vec = Array2::<f64>::zeros((n_pairs, 3));
    let mut arr_s = Array2::<i64>::zeros((n_pairs, 3));

    for (k, &src) in order.iter().enumerate() {
        arr_i[k] = out_i[src];
        arr_j[k] = out_j[src];
        arr_d[k] = out_d[src];
        arr_d_vec[[k, 0]] = out_d_vec[src][0];
        arr_d_vec[[k, 1]] = out_d_vec[src][1];
        arr_d_vec[[k, 2]] = out_d_vec[src][2];
        arr_s[[k, 0]] = out_s[src][0];
        arr_s[[k, 1]] = out_s[src][1];
        arr_s[[k, 2]] = out_s[src][2];
    }

    Ok((arr_i, arr_j, arr_d, arr_d_vec, arr_s))
}

// ── Math helpers ─────────────────────────────────────────────────────────────

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn ceil_div_usize(num: f64, den: f64) -> usize {
    (num / den).ceil() as usize
}

/// Python-compatible divmod: result satisfies `n = q*d + r` with `0 <= r < |d|`.
fn divmod_i64(n: i64, d: i64) -> (i64, i64) {
    let q = n.div_euclid(d);
    let r = n.rem_euclid(d);
    (q, r)
}

/// Multiply each row of `rows` (shape N×3) by `mat` (shape 3×3, row-major).
/// Result[i] = rows[i] @ mat  (i.e. fractional → Cartesian).
fn mat_mul_rows(rows: &Array2<f64>, mat: &ArrayView2<f64>) -> Array2<f64> {
    let n = rows.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        for j in 0..3 {
            let mut v = 0.0f64;
            for k in 0..3 {
                v += rows[[i, k]] * mat[[k, j]];
            }
            out[[i, j]] = v;
        }
    }
    out
}

/// Solve cell.T @ x.T = pos.T  →  x = pos @ inv(cell)  (fractional coords).
/// Uses Cramer's rule on the 3×3 cell matrix.
fn solve3x3(cell: &ArrayView2<f64>, pos: &Array2<f64>) -> Result<Array2<f64>, String> {
    // cell is row-major: rows are lattice vectors.
    // scaled[i] = pos[i] @ cell^{-1}
    // We need inv(cell), computed via Cramer's rule.
    let a = [cell[[0,0]], cell[[0,1]], cell[[0,2]]];
    let b = [cell[[1,0]], cell[[1,1]], cell[[1,2]]];
    let c = [cell[[2,0]], cell[[2,1]], cell[[2,2]]];

    let bxc = cross(&b, &c);
    let det = dot(&a, &bxc);

    if det.abs() < 1e-15 {
        return Err(format!(
            "Cell matrix is singular (det = {det:.3e}). Cannot compute scaled positions."
        ));
    }

    let inv_det = 1.0 / det;

    // Columns of cell^{-1} = rows of cell^{-T}.
    // inv(cell)[i][j] = cofactor[j][i] / det
    // We compute inv(cell) row by row:
    //   row 0: (b × c) / det
    //   row 1: (c × a) / det
    //   row 2: (a × b) / det
    let axb = cross(&a, &b);
    let cxa = cross(&c, &a);

    // inv_cell[row][col]
    let inv_cell = [
        [bxc[0] * inv_det, bxc[1] * inv_det, bxc[2] * inv_det],
        [cxa[0] * inv_det, cxa[1] * inv_det, cxa[2] * inv_det],
        [axb[0] * inv_det, axb[1] * inv_det, axb[2] * inv_det],
    ];

    let n = pos.nrows();
    let mut out = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        for j in 0..3 {
            // inv_cell stores M^{-T} (rows = cross-product vectors / det).
            // We need pos @ M^{-1} = pos @ (M^{-T})^T, so index as inv_cell[j][k].
            out[[i, j]] = pos[[i, 0]] * inv_cell[j][0]
                + pos[[i, 1]] * inv_cell[j][1]
                + pos[[i, 2]] * inv_cell[j][2];
        }
    }
    Ok(out)
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn simple_cell() -> ndarray::Array2<f64> {
        // Cubic 4 Å cell
        array![[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    }

    #[test]
    fn test_two_atoms_within_cutoff() {
        let pbc = [false, false, false];
        let cell = simple_cell();
        let pos = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let (i, j, d, _big_d, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 2.0, false, false, 1_000_000,
        )
        .unwrap();
        assert_eq!(i.len(), 2, "both (0→1) and (1→0) expected");
        assert!((d[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_two_atoms_outside_cutoff() {
        let pbc = [false, false, false];
        let cell = simple_cell();
        let pos = array![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let (i, _j, _d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 2.0, false, false, 1_000_000,
        )
        .unwrap();
        assert_eq!(i.len(), 0);
    }

    #[test]
    fn test_pbc_finds_image() {
        // Two atoms at 0 and 3.9 Å in a 4 Å periodic box — should find
        // the periodic image at distance 0.1 Å as well.
        let pbc = [true, false, false];
        let cell = simple_cell();
        let pos = array![[0.0, 0.0, 0.0], [3.9, 0.0, 0.0]];
        let (i, _j, d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 0.5, false, false, 1_000_000,
        )
        .unwrap();
        // Should find the image pair at ~0.1 Å.
        assert!(i.len() >= 2, "expected at least image pair; got {}", i.len());
        let min_d = d.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_d < 0.15, "image distance should be ~0.1 Å, got {min_d}");
    }

    #[test]
    fn test_self_interaction_false() {
        let pbc = [false, false, false];
        let cell = simple_cell();
        let pos = array![[0.0, 0.0, 0.0]];
        let (i, _j, _d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 5.0, false, false, 1_000_000,
        )
        .unwrap();
        assert_eq!(i.len(), 0, "no self-pair when self_interaction=false");
    }

    #[test]
    fn test_self_interaction_true() {
        let pbc = [false, false, false];
        let cell = simple_cell();
        let pos = array![[0.0, 0.0, 0.0]];
        let (i, _j, _d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 5.0, true, false, 1_000_000,
        )
        .unwrap();
        assert_eq!(i.len(), 1, "self-pair present when self_interaction=true");
    }

    #[test]
    fn test_i_sorted_ascending() {
        // FCC-like: 4 atoms
        let pbc = [true, true, true];
        let a = 3.615f64;
        let cell = array![
            [0.0, a/2.0, a/2.0],
            [a/2.0, 0.0, a/2.0],
            [a/2.0, a/2.0, 0.0]
        ];
        let pos = array![[0.0, 0.0, 0.0]];
        let (i, _j, _d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 3.0, false, false, 1_000_000,
        )
        .unwrap();
        // With one atom i must always be [0, 0, 0, ...]
        for &v in i.iter() {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_empty_input() {
        let pbc = [false, false, false];
        let cell = simple_cell();
        let pos: ndarray::Array2<f64> = ndarray::Array2::zeros((0, 3));
        let (i, _j, _d, _dd, _s) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 3.0, false, false, 1_000_000,
        )
        .unwrap();
        assert_eq!(i.len(), 0);
    }

    #[test]
    fn test_divmod_i64_positive() {
        let (q, r) = divmod_i64(7, 3);
        assert_eq!(q, 2);
        assert_eq!(r, 1);
    }

    #[test]
    fn test_divmod_i64_negative() {
        // Python: divmod(-1, 4) → (-1, 3)
        let (q, r) = divmod_i64(-1, 4);
        assert_eq!(q, -1);
        assert_eq!(r, 3);
    }

    #[test]
    fn test_shift_vector_identity() {
        // D = pos[j] - pos[i] + S @ cell  →  d = |D|
        let pbc = [true, true, true];
        let a = 3.615f64;
        let cell = array![
            [0.0, a/2.0, a/2.0],
            [a/2.0, 0.0, a/2.0],
            [a/2.0, a/2.0, 0.0]
        ];
        let pos = array![[0.0, 0.0, 0.0]];
        let (i_arr, j_arr, d_arr, big_d, s_arr) = build_neighbor_list(
            &pbc, &cell.view(), &pos.view(), 3.0, false, false, 1_000_000,
        )
        .unwrap();

        for k in 0..i_arr.len() {
            let ii = i_arr[k] as usize;
            let jj = j_arr[k] as usize;
            let s = [s_arr[[k,0]], s_arr[[k,1]], s_arr[[k,2]]];

            let d_computed = [
                pos[[jj,0]] - pos[[ii,0]]
                    + s[0] as f64 * cell[[0,0]]
                    + s[1] as f64 * cell[[1,0]]
                    + s[2] as f64 * cell[[2,0]],
                pos[[jj,1]] - pos[[ii,1]]
                    + s[0] as f64 * cell[[0,1]]
                    + s[1] as f64 * cell[[1,1]]
                    + s[2] as f64 * cell[[2,1]],
                pos[[jj,2]] - pos[[ii,2]]
                    + s[0] as f64 * cell[[0,2]]
                    + s[1] as f64 * cell[[1,2]]
                    + s[2] as f64 * cell[[2,2]],
            ];
            let d_calc = (d_computed[0].powi(2)
                + d_computed[1].powi(2)
                + d_computed[2].powi(2))
            .sqrt();

            assert!((d_arr[k] - d_calc).abs() < 1e-10,
                "d mismatch at pair {k}: stored={}, computed={d_calc}", d_arr[k]);

            for c in 0..3 {
                assert!((big_d[[k,c]] - d_computed[c]).abs() < 1e-10,
                    "D[{k},{c}] mismatch");
            }
        }
    }
}

// ── Per-atom radii and dict-cutoff extensions ─────────────────────────────────

/// Post-filter helper: keep only pairs where d < radii[i] + radii[j].
fn filter_by_radii(
    i_arr: Array1<i64>, j_arr: Array1<i64>, d_arr: Array1<f64>,
    d_vec: Array2<f64>, s_arr: Array2<i64>,
    radii: &ArrayView1<f64>,
) -> (Array1<i64>, Array1<i64>, Array1<f64>, Array2<f64>, Array2<i64>) {
    let n = i_arr.len();
    let count = (0..n)
        .filter(|&k| d_arr[k] < radii[i_arr[k] as usize] + radii[j_arr[k] as usize])
        .count();
    let mut out_i = Array1::<i64>::zeros(count);
    let mut out_j = Array1::<i64>::zeros(count);
    let mut out_d = Array1::<f64>::zeros(count);
    let mut out_dv = Array2::<f64>::zeros((count, 3));
    let mut out_s  = Array2::<i64>::zeros((count, 3));
    let mut k = 0;
    for src in 0..n {
        if d_arr[src] < radii[i_arr[src] as usize] + radii[j_arr[src] as usize] {
            out_i[k] = i_arr[src];
            out_j[k] = j_arr[src];
            out_d[k] = d_arr[src];
            for c in 0..3 { out_dv[[k,c]] = d_vec[[src,c]]; out_s[[k,c]] = s_arr[[src,c]]; }
            k += 1;
        }
    }
    (out_i, out_j, out_d, out_dv, out_s)
}

/// Neighbor list with per-atom radii cutoffs.
/// max_cutoff = 2 * max(radii); post-filter keeps d < radii[i] + radii[j].
pub fn build_neighbor_list_radii(
    pbc: &[bool; 3],
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    radii: &ArrayView1<f64>,
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> Result<(Array1<i64>, Array1<i64>, Array1<f64>, Array2<f64>, Array2<i64>), String> {
    if radii.is_empty() {
        return Ok((Array1::zeros(0), Array1::zeros(0), Array1::zeros(0),
                   Array2::zeros((0,3)), Array2::zeros((0,3))));
    }
    let max_r = radii.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_cutoff = 2.0 * max_r;
    let (i_arr, j_arr, d_arr, d_vec, s_arr) = build_neighbor_list(
        pbc, cell, positions, max_cutoff, self_interaction, use_scaled_positions, max_nbins,
    )?;
    Ok(filter_by_radii(i_arr, j_arr, d_arr, d_vec, s_arr, radii))
}

/// Neighbor list with dict-style (element-pair) cutoffs.
/// max_cutoff = max(cutoff_vals); post-filter uses per-pair lookup.
/// cutoff keys are (zi_keys[k], zj_keys[k]) -> cutoff_vals[k]; symmetric pairs included.
pub fn build_neighbor_list_dict(
    pbc: &[bool; 3],
    cell: &ArrayView2<f64>,
    positions: &ArrayView2<f64>,
    numbers: &ArrayView1<i64>,
    zi_keys: &[i64],
    zj_keys: &[i64],
    cutoff_vals_arr: &[f64],
    self_interaction: bool,
    use_scaled_positions: bool,
    max_nbins: usize,
) -> Result<(Array1<i64>, Array1<i64>, Array1<f64>, Array2<f64>, Array2<i64>), String> {
    if cutoff_vals_arr.is_empty() {
        return Ok((Array1::zeros(0), Array1::zeros(0), Array1::zeros(0),
                   Array2::zeros((0,3)), Array2::zeros((0,3))));
    }
    let max_cutoff = cutoff_vals_arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (i_arr, j_arr, d_arr, d_vec, s_arr) = build_neighbor_list(
        pbc, cell, positions, max_cutoff, self_interaction, use_scaled_positions, max_nbins,
    )?;

    // Build lookup: both (zi,zj) and (zj,zi) -> cutoff
    let mut table: HashMap<(i64, i64), f64> = HashMap::new();
    for idx in 0..zi_keys.len() {
        let zi = zi_keys[idx];
        let zj = zj_keys[idx];
        let c  = cutoff_vals_arr[idx];
        table.insert((zi, zj), c);
        table.insert((zj, zi), c); // symmetric
    }

    let n = i_arr.len();
    let count = (0..n).filter(|&k| {
        let zi = numbers[i_arr[k] as usize];
        let zj = numbers[j_arr[k] as usize];
        table.get(&(zi, zj)).map_or(false, |&c| d_arr[k] < c)
    }).count();

    let mut out_i = Array1::<i64>::zeros(count);
    let mut out_j = Array1::<i64>::zeros(count);
    let mut out_d = Array1::<f64>::zeros(count);
    let mut out_dv = Array2::<f64>::zeros((count, 3));
    let mut out_s  = Array2::<i64>::zeros((count, 3));
    let mut k = 0;
    for src in 0..n {
        let zi = numbers[i_arr[src] as usize];
        let zj = numbers[j_arr[src] as usize];
        if table.get(&(zi, zj)).map_or(false, |&c| d_arr[src] < c) {
            out_i[k] = i_arr[src];
            out_j[k] = j_arr[src];
            out_d[k] = d_arr[src];
            for c in 0..3 { out_dv[[k,c]] = d_vec[[src,c]]; out_s[[k,c]] = s_arr[[src,c]]; }
            k += 1;
        }
    }
    Ok((out_i, out_j, out_d, out_dv, out_s))
}
