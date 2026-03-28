use pyo3::prelude::*;

mod minkowski;
mod supercell;

#[pymodule]
fn _geometry_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minkowski::minkowski_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(supercell::make_supercell_positions_rs, m)?)?;
    Ok(())
}
