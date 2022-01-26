use ndarray_linalg::svd::SVD;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::pyfunction;
use pyo3::{PyResult, Python};

#[pyfunction]
pub fn nclr_polar<'py>(
    py: Python<'py>,
    m: PyReadonlyArray2<f64>,
) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
    let (r, s) = polar(&m.as_array().to_owned());
    Ok((r.into_pyarray(py), s.into_pyarray(py)))
}

pub fn polar(m: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (u, sigma, vt) = m.svd(true, true).unwrap();

    let r = u.as_ref().unwrap().dot(vt.as_ref().unwrap());

    let full_sig = Array2::from_diag(&sigma);

    let s = (vt.as_ref().unwrap().t().dot(&full_sig)).dot(&vt.unwrap());
    (r, s)
}
