//! Minkowski basis reduction — Rust port of ASE's minkowski_reduction.py
//!
//! Implements `reduction_full` (3D case) and the supporting routines
//! `reduction_gauss`, `relevant_vectors_2D`, `closest_vector`.
//! The algorithm and convergence logic match the Python exactly so that
//! `np.allclose(rust_result, python_result)` holds for any valid cell.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

const MAX_IT: usize = 100_000;

// ── CycleChecker ─────────────────────────────────────────────────────────────

/// Cycle detector for 3D reduction.  Detects if we visit the same H matrix twice.
/// Equivalent to Python's CycleChecker(d=3).
struct CycleChecker3D {
    max_len: usize,
    buf: Vec<[i64; 9]>,
    head: usize,
    count: usize,
}

impl CycleChecker3D {
    fn new() -> Self {
        // d=3: n=12, max_len = prod([12,11,10]) * 3 = 3960
        Self { max_len: 3960, buf: vec![[0i64; 9]; 3960], head: 0, count: 0 }
    }

    /// Returns true if H was already seen (cycle detected), then records H.
    fn add_site(&mut self, h: &[[i64; 3]; 3]) -> bool {
        let flat: [i64; 9] = [
            h[0][0], h[0][1], h[0][2],
            h[1][0], h[1][1], h[1][2],
            h[2][0], h[2][1], h[2][2],
        ];
        // Check all stored entries (including zero-initialized ones, matching Python)
        let found = self.buf.iter().any(|v| v == &flat);
        // Rolling insert at head (Python: np.roll + assign [0])
        self.buf[self.head] = flat;
        self.head = (self.head + 1) % self.max_len;
        if self.count < self.max_len { self.count += 1; }
        found
    }
}

/// Cycle detector for 2D Gauss reduction.
/// Equivalent to Python's CycleChecker(d=2).
struct CycleChecker2D {
    max_len: usize,
    buf: Vec<[i64; 6]>,
    head: usize,
    count: usize,
}

impl CycleChecker2D {
    fn new() -> Self {
        // d=2: n=6, max_len = prod([6,5]) * 2 = 60
        Self { max_len: 60, buf: vec![[0i64; 6]; 60], head: 0, count: 0 }
    }

    fn add_site(&mut self, site: &[[i64; 3]; 2]) -> bool {
        let flat: [i64; 6] = [
            site[0][0], site[0][1], site[0][2],
            site[1][0], site[1][1], site[1][2],
        ];
        let found = self.buf.iter().any(|v| v == &flat);
        self.buf[self.head] = flat;
        self.head = (self.head + 1) % self.max_len;
        if self.count < self.max_len { self.count += 1; }
        found
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

#[inline]
fn dot2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    a[0]*b[0] + a[1]*b[1]
}

#[inline]
fn norm3(a: &[f64; 3]) -> f64 { dot3(a, a).sqrt() }

/// Compute u = h @ B  where h is an integer row-vector and B is 3×3 float.
/// Equivalent to: u[j] = sum_k h[k] * B[k][j]
#[inline]
fn row_mat_mul(h: &[i64; 3], b: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        h[0] as f64 * b[0][0] + h[1] as f64 * b[1][0] + h[2] as f64 * b[2][0],
        h[0] as f64 * b[0][1] + h[1] as f64 * b[1][1] + h[2] as f64 * b[2][1],
        h[0] as f64 * b[0][2] + h[1] as f64 * b[1][2] + h[2] as f64 * b[2][2],
    ]
}

/// Compute R = H @ B  where H is 3×3 integer and B is 3×3 float.
fn mat_mul_int_float(h: &[[i64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += h[i][k] as f64 * b[k][j];
            }
        }
    }
    r
}

// ── Core algorithm ────────────────────────────────────────────────────────────

/// Find 7 shortest 2D vectors from combinations {-1,0,1}² of u and v.
/// Returns (vs[7], cs[7]) where vs[k] = cs[k][0]*u + cs[k][1]*v.
fn relevant_vectors_2d(u: &[f64; 2], v: &[f64; 2]) -> ([[f64; 2]; 7], [[i64; 2]; 7]) {
    // All 9 combinations of {-1,0,1}^2 in column-major order matching Python's
    // itertools.product([-1,0,1], repeat=2)
    let cs: [[i64; 2]; 9] = [
        [-1,-1], [-1,0], [-1,1],
        [ 0,-1], [ 0,0], [ 0,1],
        [ 1,-1], [ 1,0], [ 1,1],
    ];
    let vs: [[f64; 2]; 9] = core::array::from_fn(|k| {
        let c0 = cs[k][0] as f64;
        let c1 = cs[k][1] as f64;
        [c0*u[0] + c1*v[0], c0*u[1] + c1*v[1]]
    });
    // Sort indices by norm
    let mut idx: [usize; 9] = [0,1,2,3,4,5,6,7,8];
    idx.sort_by(|&a, &b| {
        let na = vs[a][0]*vs[a][0] + vs[a][1]*vs[a][1];
        let nb = vs[b][0]*vs[b][0] + vs[b][1]*vs[b][1];
        na.partial_cmp(&nb).unwrap()
    });
    let mut out_vs = [[0.0f64; 2]; 7];
    let mut out_cs = [[0i64; 2]; 7];
    for i in 0..7 {
        out_vs[i] = vs[idx[i]];
        out_cs[i] = cs[idx[i]];
    }
    (out_vs, out_cs)
}

/// Find the closest lattice vector to t0 in the 2D lattice spanned by u, v.
/// Returns integer coefficients a such that a[0]*u + a[1]*v is nearest to -t0.
/// Matches Python's closest_vector(t0, u, v).
fn closest_vector(t0: &[f64; 2], u: &[f64; 2], v: &[f64; 2]) -> [i64; 2] {
    let mut t = *t0;
    let mut a = [0i64; 2];
    let (rs, cs) = relevant_vectors_2d(u, v);

    let mut dprev = f64::INFINITY;
    for _ in 0..MAX_IT {
        // ds[k] = |rs[k] + t|
        let mut min_d = f64::INFINITY;
        let mut min_idx = 0usize;
        for k in 0..7 {
            let dx = rs[k][0] + t[0];
            let dy = rs[k][1] + t[1];
            let d = (dx*dx + dy*dy).sqrt();
            if d < min_d { min_d = d; min_idx = k; }
        }

        if min_idx == 0 || min_d >= dprev {
            return a;
        }
        dprev = min_d;
        let r = &rs[min_idx];
        let kopt = (-dot2(&t, r) / dot2(r, r)).round() as i64;
        a[0] += kopt * cs[min_idx][0];
        a[1] += kopt * cs[min_idx][1];
        t[0] = t0[0] + a[0] as f64 * u[0] + a[1] as f64 * v[0];
        t[1] = t0[1] + a[0] as f64 * u[1] + a[1] as f64 * v[1];
    }
    // In practice MAX_IT is never reached for valid cells
    a
}

/// Gauss-reduce two lattice basis vectors.
/// Matches Python's reduction_gauss(B, hu, hv) exactly.
/// Returns (hu_out, hv_out) = (hv, hu) at convergence.
fn reduction_gauss(b: &[[f64; 3]; 3], hu_in: &[i64; 3], hv_in: &[i64; 3])
    -> ([i64; 3], [i64; 3])
{
    let mut cycle_checker = CycleChecker2D::new();
    let mut hu = *hu_in;
    let mut hv = *hv_in;

    for _ in 0..MAX_IT {
        let u = row_mat_mul(&hu, b);
        let v = row_mat_mul(&hv, b);
        let x = (dot3(&u, &v) / dot3(&u, &u)).round() as i64;
        let new_hu = [hv[0] - x*hu[0], hv[1] - x*hu[1], hv[2] - x*hu[2]];
        let new_hv = hu;
        hu = new_hu;
        hv = new_hv;
        let u = row_mat_mul(&hu, b);
        let v = row_mat_mul(&hv, b);
        let site = [hu, hv];
        if dot3(&u, &u) >= dot3(&v, &v) || cycle_checker.add_site(&site) {
            return (hv, hu);
        }
    }
    // Convergence guaranteed for valid lattice bases; panic only in degenerate cases
    (hv, hu)
}

/// Full 3D Minkowski reduction.
/// Returns (R, H) where R = H @ B is the reduced cell (same as Python's reduction_full).
pub fn reduction_full(b: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[i64; 3]; 3]) {
    let mut cycle_checker = CycleChecker3D::new();
    let mut h: [[i64; 3]; 3] = [[1,0,0],[0,1,0],[0,0,1]];
    let mut norms = [norm3(&b[0]), norm3(&b[1]), norm3(&b[2])];

    for _ in 0..MAX_IT {
        // Sort H rows by norms (stable, matching Python's kind='merge')
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &bi| norms[a].partial_cmp(&norms[bi]).unwrap_or(std::cmp::Ordering::Equal));
        h = [h[order[0]], h[order[1]], h[order[2]]];
        let _ = [norms[order[0]], norms[order[1]], norms[order[2]]]; // intermediate sort (overwritten)

        let hw = h[2];
        let (hu, hv) = reduction_gauss(b, &h[0], &h[1]);
        h = [hu, hv, hw];

        // R = H @ B
        let r = mat_mul_int_float(&h, b);

        // Orthogonalize first two vectors (Gram-Schmidt)
        let u = &r[0];
        let v = &r[1];
        let u_norm = norm3(u);
        let x_hat = [u[0]/u_norm, u[1]/u_norm, u[2]/u_norm];
        let dot_vx = dot3(v, &x_hat);
        let mut y = [v[0] - x_hat[0]*dot_vx, v[1] - x_hat[1]*dot_vx, v[2] - x_hat[2]*dot_vx];
        let y_norm = norm3(&y);
        y = [y[0]/y_norm, y[1]/y_norm, y[2]/y_norm];

        // Project R rows onto 2D subspace [X, Y]: pu[k] = (dot(R[k], X), dot(R[k], Y))
        let pu = [dot3(&r[0], &x_hat), dot3(&r[0], &y)];
        let pv = [dot3(&r[1], &x_hat), dot3(&r[1], &y)];
        let pw = [dot3(&r[2], &x_hat), dot3(&r[2], &y)];

        let nb = closest_vector(&pw, &pu, &pv);

        // H[2] = [nb[0], nb[1], 1] @ H  (linear combination of H rows)
        h[2] = [
            nb[0]*h[0][0] + nb[1]*h[1][0] + h[2][0],
            nb[0]*h[0][1] + nb[1]*h[1][1] + h[2][1],
            nb[0]*h[0][2] + nb[1]*h[1][2] + h[2][2],
        ];

        // Recompute R and norms
        let r = mat_mul_int_float(&h, b);
        norms = [norm3(&r[0]), norm3(&r[1]), norm3(&r[2])];

        if norms[2] >= norms[1] || cycle_checker.add_site(&h) {
            return (r, h);
        }
    }
    // Should not reach here for valid lattice basis
    let r = mat_mul_int_float(&h, b);
    (r, h)
}

// ── PyO3 entry point ──────────────────────────────────────────────────────────

/// Minkowski-reduce a 3×3 lattice cell (3D full reduction only).
///
/// Equivalent to the `dim==3` branch of Python's `minkowski_reduce(cell, pbc=True)`.
/// Returns (rcell, op) where rcell = op @ cell and op is the unimodular
/// transformation matrix.  Handedness maintenance is left to the Python caller.
///
/// Parameters
/// ----------
/// cell : (3, 3) float64 array — lattice vectors (row-major)
///
/// Returns
/// -------
/// (rcell, op) : ((3,3) float64, (3,3) int64)
#[pyfunction]
pub fn minkowski_reduce_rs<'py>(
    py: Python<'py>,
    cell: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<i64>>)> {
    let c = cell.as_array();
    if c.shape() != [3, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cell must be shape (3, 3)"
        ));
    }
    let b: [[f64; 3]; 3] = [
        [c[[0,0]], c[[0,1]], c[[0,2]]],
        [c[[1,0]], c[[1,1]], c[[1,2]]],
        [c[[2,0]], c[[2,1]], c[[2,2]]],
    ];

    let (rcell, op) = reduction_full(&b);

    // Pack into ndarray outputs
    let mut arr_r = Array2::<f64>::zeros((3, 3));
    let mut arr_op = Array2::<i64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            arr_r[[i, j]]  = rcell[i][j];
            arr_op[[i, j]] = op[i][j];
        }
    }

    Ok((arr_r.into_pyarray(py).into(), arr_op.into_pyarray(py).into()))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fcc_cell() -> [[f64; 3]; 3] {
        // FCC conventional cell, a=3.615 Å
        let a = 3.615f64;
        [[0.0, a/2.0, a/2.0],
         [a/2.0, 0.0, a/2.0],
         [a/2.0, a/2.0, 0.0]]
    }

    fn cubic_cell() -> [[f64; 3]; 3] {
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    }

    #[test]
    fn test_reduction_full_identity_already_reduced() {
        // Cubic cell is already Minkowski-reduced (equal lengths, orthogonal)
        let b = cubic_cell();
        let (_r, h) = reduction_full(&b);
        // H should be identity (or equivalent unimodular transform)
        let det = h[0][0]*(h[1][1]*h[2][2]-h[1][2]*h[2][1])
                - h[0][1]*(h[1][0]*h[2][2]-h[1][2]*h[2][0])
                + h[0][2]*(h[1][0]*h[2][1]-h[1][1]*h[2][0]);
        assert!(det == 1 || det == -1, "H must be unimodular");
    }

    #[test]
    fn test_reduction_full_fcc_primitive() {
        // FCC primitive cell — already reduced
        let b = fcc_cell();
        let (r, h) = reduction_full(&b);
        let det = h[0][0]*(h[1][1]*h[2][2]-h[1][2]*h[2][1])
                - h[0][1]*(h[1][0]*h[2][2]-h[1][2]*h[2][0])
                + h[0][2]*(h[1][0]*h[2][1]-h[1][1]*h[2][0]);
        assert!(det == 1 || det == -1, "H must be unimodular");
        // All norms should be equal for FCC primitive cell
        let n0 = norm3(&r[0]);
        let n1 = norm3(&r[1]);
        let n2 = norm3(&r[2]);
        assert!((n0 - n1).abs() < 1e-10);
        assert!((n1 - n2).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_full_returns_shorter_vectors() {
        // A deliberately distorted cell: one long vector
        let b = [[10.0f64, 7.0, 3.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]];
        let (r, h) = reduction_full(&b);
        // norms of R must be <= norms of B (sorted)
        let mut nb = [norm3(&b[0]), norm3(&b[1]), norm3(&b[2])];
        let mut nr = [norm3(&r[0]), norm3(&r[1]), norm3(&r[2])];
        nb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        nr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..3 {
            assert!(nr[i] <= nb[i] + 1e-10,
                "reduced norm {} > original norm {}", nr[i], nb[i]);
        }
        let det = h[0][0]*(h[1][1]*h[2][2]-h[1][2]*h[2][1])
                - h[0][1]*(h[1][0]*h[2][2]-h[1][2]*h[2][0])
                + h[0][2]*(h[1][0]*h[2][1]-h[1][1]*h[2][0]);
        assert!(det == 1 || det == -1, "H unimodular, got det={}", det);
    }

    #[test]
    fn test_reduction_gauss_reduces_norms() {
        let b = [[3.0f64, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
        let hu = [1i64, 0, 0];
        let hv = [1i64, 1, 0]; // not reduced
        let (hu_out, hv_out) = reduction_gauss(&b, &hu, &hv);
        let u = row_mat_mul(&hu_out, &b);
        let v = row_mat_mul(&hv_out, &b);
        assert!(dot3(&u, &u) <= dot3(&v, &v) + 1e-10,
            "first returned vector should be shorter");
    }
}
