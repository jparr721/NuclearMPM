pub mod linear_algebra;

use linear_algebra::__pyo3_get_function_nclr_polar;
use ndarray::parallel::prelude::*;
use ndarray_linalg::solve::Determinant;
use num::clamp;
use numpy::ndarray::{Array1, Array2, Array3, Array4, Axis};
use numpy::{IntoPyArray, PyArray2, PyArray4, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

fn constant_hardening(mu: f64, lambda: f64, e: f64) -> (f64, f64) {
    (mu * e, lambda * e)
}

fn snow_hardening(mu: f64, lambda: f64, h: f64, jp: f64) -> (f64, f64) {
    let e = (h * (1.0 - jp)).exp();
    constant_hardening(mu, lambda, e)
}

fn fixed_corotated_stress(
    f: &Array2<f64>,
    inv_dx: f64,
    mu: f64,
    lambda: f64,
    dt: f64,
    volume: f64,
    mass: f64,
    c: &Array2<f64>,
) -> Array2<f64> {
    let jacobian = f.det().unwrap();
    let (r, _) = linear_algebra::polar(&f);
    let d_inv = 4.0 * inv_dx * inv_dx;

    let cauchy = (2.0 * mu * (f - r)).dot(&f.t()) + lambda * (jacobian - 1.0) * jacobian;
    let stress = -(dt * volume) * (d_inv * cauchy);
    stress + mass * c
}

fn grid_op(
    resolution: usize,
    boundary: usize,
    dx: f64,
    dt: f64,
    gravity: f64,
    velocity: &mut Array4<f64>,
    mass: &Array4<f64>,
) {
    let v_allowed = dx * 0.9 / dt;

    //     velocity
    //         .axis_iter_mut(Axis(0))
    //         .into_par_iter()
    //         .enumerate()
    //         .for_each(|(i, vel2)| {
    //             vel2.axis_iter(Axis(0)).enumerate().for_each(|(j, vel3)| {
    //                 vel3.axis_iter(Axis(0)).enumerate().for_each(|(k, value)| {
    //                     if mass[[i, j, k, 0]] > 0.0 {
    //                         *value /= mass[[i, j, k, 0]];
    //                         value += dt * gravity;
    //                         velocity = clamp(velocity, -v_allowed, v_allowed);
    //                     }
    //                 })
    //             })
    //         });

    for i in 0..resolution + 1 {
        for j in 0..resolution + 1 {
            for k in 0..resolution + 1 {
                if mass[[i, j, k, 0]] > 0.0 {
                    velocity[[i, j, k, 0]] /= mass[[i, j, k, 0]];
                    velocity[[i, j, k, 1]] /= mass[[i, j, k, 0]];
                    velocity[[i, j, k, 2]] /= mass[[i, j, k, 0]];

                    velocity[[i, j, k, 1]] += dt * gravity;

                    velocity[[i, j, k, 0]] = clamp(velocity[[i, j, k, 0]], -v_allowed, v_allowed);
                    velocity[[i, j, k, 1]] = clamp(velocity[[i, j, k, 1]], -v_allowed, v_allowed);
                    velocity[[i, j, k, 2]] = clamp(velocity[[i, j, k, 2]], -v_allowed, v_allowed);
                }

                [i, j, k].iter().enumerate().for_each(|(index, val)| {
                    if val < &boundary && velocity[[i, j, k, index]] < 0.0 {
                        velocity[[i, j, k, index]] = 0.0
                    }

                    if val >= &(resolution - boundary) && velocity[[i, j, k, index]] > 0.0 {
                        velocity[[i, j, k, index]] = 0.0
                    }
                })
            }
        }
    }
}

fn p2g(
    inv_dx: f64,
    hardening: f64,
    mu_0: f64,
    lambda_0: f64,
    mass: f64,
    dx: f64,
    dt: f64,
    volume: f64,
    grid_velocity: &mut Array3<f64>,
    grid_mass: &mut Array3<f64>,
    x: Array2<f64>,
    v: Array2<f64>,
    f: Array3<f64>,
    c: Array3<f64>,
    jp: Array1<f64>,
    model: usize,
) {
    x.axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(p, pos)| {
            let base_coord = pos.to_owned() * inv_dx - 0.5;
            let fx = (pos.to_owned() * inv_dx) - &base_coord;
            let base_coordi = base_coord.mapv(|e| e as i64);
            let w_i = (1.5 - &fx).mapv(|v| v.powi(2)) * 0.5;
            let w_j = (&fx - 1.0).mapv(|v| v.powi(2)) - 0.75;
            let w_k = (&fx - 0.5).mapv(|v| v.powi(2)) * 0.5;

            let (mu, lambda) = if model == 1 {
                constant_hardening(mu_0, lambda_0, hardening)
            } else {
                snow_hardening(mu_0, lambda_0, hardening, jp[p])
            };
        });
}

#[pyfunction]
pub fn nclr_grid_op<'py>(
    py: Python<'py>,
    resolution: usize,
    boundary: usize,
    dx: f64,
    dt: f64,
    gravity: f64,
    velocity: PyReadonlyArray4<f64>,
    mass: PyReadonlyArray4<f64>,
) -> PyResult<(&'py PyArray4<f64>, &'py PyArray4<f64>)> {
    let mut v = velocity.as_array().to_owned();
    let m = mass.as_array().to_owned();
    grid_op(
        resolution,
        boundary,
        dx,
        dt,
        gravity,
        &mut v,
        &mass.as_array().to_owned(),
    );

    Ok((v.into_pyarray(py), m.into_pyarray(py)))
}

#[pyfunction]
pub fn nclr_stress<'py>(
    py: Python<'py>,
    f: PyReadonlyArray2<f64>,
    inv_dx: f64,
    mu: f64,
    lambda: f64,
    dt: f64,
    volume: f64,
    mass: f64,
    c: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let stress = fixed_corotated_stress(
        &f.as_array().to_owned(),
        inv_dx,
        mu,
        lambda,
        dt,
        volume,
        mass,
        &c.as_array().to_owned(),
    );
    Ok(stress.into_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn nuclear_mpm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nclr_polar, m)?)?;
    m.add_function(wrap_pyfunction!(nclr_stress, m)?)?;
    m.add_function(wrap_pyfunction!(nclr_grid_op, m)?)?;
    Ok(())
}
