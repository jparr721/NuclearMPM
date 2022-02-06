#include "math.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>
#include <vector>

struct Particle {
    // Position and velocity
    Vec x, v;
    // Deformation gradient
    Mat F;
    // Affine momentum from APIC
    Mat C;
    // Determinant of the deformation gradient (i.e. volume)
    nc_real Jp;
    // Color
    int c;

    Particle(Vec x, int c, Vec v = constvec(0)) : x(x), v(v), F(diag(1)), C(constmat(0)), Jp(1), c(c) {}
};

struct Cell {
    Vec velocity;
    double mass;
    Cell() : velocity(constvec(0)), mass(0.0) {}
};

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

inline auto nclr_svd(const Eigen::Matrix3d &a, Eigen::Matrix3d &U, Eigen::Matrix3d &sig, Eigen::Matrix3d &V) -> void {
    const auto svd = Eigen::JacobiSVD<Eigen::Matrix3d>(a, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    V = svd.matrixV();
    const auto values = svd.singularValues();
    sig(0, 0) = values(0);
    sig(1, 1) = values(1);
    sig(2, 2) = values(2);
}

inline auto nclr_polar(const Eigen::Matrix3d &m, Eigen::Matrix3d &R, Eigen::Matrix3d &S) -> void {
    Eigen::Matrix3d sig;
    Eigen::Matrix3d U, V;
    nclr_svd(m, U, sig, V);

    R = U * V.transpose();
    S = V * sig * V.transpose();
}

// Hardening coefficients
auto nclr_constant_hardening(const double mu, const double lambda, const double e) -> std::pair<double, double>;
auto nclr_snow_hardening(const double mu, const double lambda, const double h, const double Jp)
        -> std::pair<double, double>;

/* // Cauchy stress */
/* auto nclr_fixed_corotated_stress(const Eigen::Matrix3d &F, const double inv_dx, const double mu, const double lambda, */
/*                                  const double dt, const double volume, const double mass, const Eigen::Matrix3d &C) */
/*         -> Eigen::MatrixXd; */

/* // MPM Operations */
/* auto nclr_p2g(const int res, const double inv_dx, const double hardening, const double mu_0, const double lambda_0, */
/*               const double mass, const double dx, const double dt, const double volume, */
/*               std::vector<Eigen::Vector3d> &grid_velocity, std::vector<double> &grid_mass, */
/*               std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F, */
/*               std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp) -> void; */
auto nclr_grid_op(float gravity, std::vector<Cell> &cells) -> void;
/* auto nclr_g2p(const int res, const double inv_dx, const double dt, const std::vector<Eigen::Vector3d> &grid_velocity, */
/*               std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F, */
/*               std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp) -> void; */
/* auto nclr_mpm(const double inv_dx, const double hardening, const double mu_0, const double lambda_0, const double mass, */
/*               const double dx, const double dt, const double volume, const unsigned int res, const double gravity, */
/*               const std::size_t timesteps, const Eigen::MatrixX3d &x) -> std::vector<Eigen::MatrixXd>; */
