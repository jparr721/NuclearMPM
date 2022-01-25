use ndarray_linalg::convert::transpose_data;
use ndarray_linalg::svd::SVD;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::pyfunction;
use pyo3::{exceptions::PyRuntimeError, PyErr, PyResult, Python};

#[pyfunction]
pub fn polar_decomposition<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<f64>,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
    let (u, sigma, vt) = m.as_array().svd(true, true).unwrap();
    let r = u.as_ref().unwrap() * vt.as_ref().unwrap();

    let s = (vt.as_ref().unwrap().t() * sigma.view()).dot(vt.unwrap());
    Ok((r.into_pyarray(py), s.into_pyarray(py)))
}
