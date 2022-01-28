#include "sifakis_svd.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

enum class MaterialModel {
    kNeoHookean = 0x00,
    kSnow = 0x01,
};

inline auto diagonal(const double value) -> Eigen::Matrix3d {
    const Eigen::Vector3d diagonal = Eigen::Vector3d::Constant(value);
    return diagonal.array().matrix().asDiagonal();
}

inline auto vec_to_mat(const std::vector<Eigen::Vector3d> &data) -> Eigen::MatrixXd {
    Eigen::MatrixX3d mat;
    mat.resize(data.size(), 3);
    for (int i = 0; i < data.size(); ++i) { mat.row(i) = data.at(i); }
    return mat;
}

// Utils
inline auto oob(const Eigen::Vector3i base, const int res, const Eigen::Vector3i ijk = Eigen::Vector3i::Zero())
        -> bool {
    const Eigen::Vector3i bijk = base + ijk;
    const Eigen::Vector3i comp = Eigen::Vector3i::Ones() * res;

    for (int ii = 0; ii < 3; ++ii) {
        if (bijk(ii) >= comp(ii) || bijk(ii) < 0) { return true; }
    }
    return false;
}

inline auto nclr_polar(const Eigen::Matrix3d &m) -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {
    Eigen::Vector3d sig;
    Eigen::Matrix3d U, V;
    nclr_svd(m, U, sig, V);

    const Eigen::Matrix3d R = U * V.transpose();
    const Eigen::Matrix3d S = (V.transpose().array() * sig.replicate(1, 3).transpose().array()).matrix() * V;

    return std::make_pair(R, S);
}

inline auto sqr(const Eigen::Vector3d &a) -> Eigen::Vector3d { return a.array().square().matrix(); }

// Hardening coefficients
auto nclr_constant_hardening(const double mu, const double lambda, const double e) -> std::pair<double, double>;
auto nclr_snow_hardening(const double mu, const double lambda, const double h, const double Jp)
        -> std::pair<double, double>;

// Cauchy stress
auto nclr_fixed_corotated_stress(const Eigen::Matrix3d &F, const double inv_dx, const double mu, const double lambda,
                                 const double dt, const double volume, const double mass, const Eigen::Matrix3d &C)
        -> Eigen::MatrixXd;

// MPM Operations
auto nclr_p2g(const int res, const double inv_dx, const double hardening, const double mu_0, const double lambda_0,
              const double mass, const double dx, const double dt, const double volume,
              std::vector<Eigen::Vector3d> &grid_velocity, std::vector<double> &grid_mass,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void;
auto nclr_grid_op(const int res, const double dx, const double dt, const double gravity,
                  const std::vector<double> &grid_mass, std::vector<Eigen::Vector3d> &grid_velocity) -> void;
auto nclr_g2p(const int res, const double inv_dx, const double dt, const std::vector<Eigen::Vector3d> &grid_velocity,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void;
auto nclr_mpm(const double inv_dx, const double hardening, const double mu_0, const double lambda_0, const double mass,
              const double dx, const double dt, const double volume, const unsigned int res, const double gravity,
              const std::size_t timesteps, const Eigen::MatrixX3d &x) -> std::vector<Eigen::MatrixXd>;
