use pyo3::prelude::*;

mod vasp;
mod xyz;

#[pymodule]
fn _io_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vasp::parse_poscar_positions_rs, m)?)?;
    m.add_function(wrap_pyfunction!(vasp::format_poscar_positions_rs, m)?)?;
    m.add_function(wrap_pyfunction!(vasp::parse_xdatcar_coords_rs, m)?)?;
    m.add_function(wrap_pyfunction!(vasp::format_xdatcar_config_rs, m)?)?;
    m.add_function(wrap_pyfunction!(xyz::parse_xyz_block_rs, m)?)?;
    m.add_function(wrap_pyfunction!(xyz::format_xyz_block_rs, m)?)?;
    Ok(())
}
