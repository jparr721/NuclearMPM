pub mod linear_algebra;

use linear_algebra::{__pyo3_get_function_polar_decomposition, polar_decomposition};
use ndarray_linalg::convert::transpose_data;
use ndarray_linalg::solve::Determinant;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyErr, PyResult, Python};

fn constant_hardening(mu: f64, lambda: f64, e: f64) -> (f64, f64) {
    (mu * e, lambda * e)
}

fn snow_hardening(mu: f64, lambda: f64, h: f64, jp: f64) -> (f64, f64) {
    let e = (h * (1.0 - jp)).exp();
    constant_hardening(mu, lambda, e)
}

// fn fixed_corotated_stress(
//     f: PyReadonlyArray2<f64>,
//     inv_dx: f64,
//     mu: f64,
//     lambda: f64,
//     dt: f64,
//     volume: f64,
//     mass: f64,
//     c: PyReadonlyArray2<f64>,
// ) -> PyArray2<f64> {
//     let jacobian = f.as_array().det().unwrap();
//     let (r, _) = linear_algebra::polar_decomposition(f).unwrap();
//     let d_inv = 4 * inv_dx * inv_dx;
//     let cauchy = (2 * mu * (f - r) * transpose_data(f)) + lambda * (jacobian - 1) * jacobian;
// }

/// Formats the sum of two numbers as string.
#[pyfunction]
fn det(_py: Python, m: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let y = m.as_array().det().unwrap();
    Ok(y)
}

/// A Python module implemented in Rust.
#[pymodule]
fn nuclear_mpm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(det, m)?)?;
    m.add_function(wrap_pyfunction!(polar_decomposition, m)?)?;
    Ok(())
}
