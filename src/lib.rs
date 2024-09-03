mod woe;
use pyo3::types::PyModule;
use pyo3::{pymodule, Bound, PyResult};

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
