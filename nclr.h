#include "sifakis_svd.h"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

template<typename T>
auto clamp(T value, T min, T max) -> T {
    if (value < min) {
        return min;
    } else if (value > max) {
        return max;
    } else {
        return value;
    }
}

template<typename T>
inline auto vec_to_mat(const std::vector<std::vector<T>> &data) -> Eigen::MatrixXd {
    Eigen::Matrix<T, -1, -1> mat(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) mat.row(i) = Eigen::Matrix<T, -1, 1>::Map(&data[i][0], data[0].size());
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

inline auto polar(const Eigen::Matrix3d &m) -> std::pair<Eigen::Matrix3d, Eigen::Matrix3d> {}

// Hardening coefficients
auto nclr_constant_hardening(const double mu, const double lambda, const double e) -> std::pair<double, double>;
auto nclr_snow_hardening(const double mu, const double lambda, const double h, const double Jp)
        -> std::pair<double, double>;

// Cauchy stress
auto nclr_fixed_corotated_stress(const Eigen::Matrix3d &F, const double inv_dx, const double mu, const double lambda,
                                 const double dt, const double volume, const double mass, const Eigen::Matrix3d &C)
        -> Eigen::MatrixXd;

// MPM Operations
auto nclr_p2g(const double inv_dx, const double hardening, const double mu, const double lambda, const double mass,
              const double dx, const double dt, const double volume, Eigen::Tensor<double, 4> &grid_velocity,
              Eigen::Tensor<double, 4> &grid_mass, std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v,
              std::vector<Eigen::Matrix3d> &F, std::vector<Eigen::Matrix3d> &C, Eigen::VectorXi, int model) -> void;
auto nclr_grid_op(const int grid_resolution, const double dx, const double dt, const double gravity,
                  Eigen::Tensor<double, 4> &grid_velocity, Eigen::Tensor<double, 4> &grid_mass) -> void;
auto nclr_g2p(const double inv_dx, const double dt, Eigen::Tensor<double, 4> &grid_velocity,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, Eigen::VectorXi, int model) -> void;
