#include "nclr.h"
#include <Eigen/SVD>
#include <cmath>

namespace py = pybind11;

auto nclr_constant_hardening(const double mu, const double lambda, const double e) -> std::pair<double, double> {
    return std::make_pair(mu * e, lambda * e);
}

auto nclr_snow_hardening(const double mu, const double lambda, const double h, const double jp)
        -> std::pair<double, double> {
    const double e = std::exp(h * (1.0 - jp));
    return nclr_constant_hardening(mu, lambda, e);
}


// Cauchy stress
auto nclr_fixed_corotated_stress(const Eigen::Matrix3d &F, const double inv_dx, const double mu, const double lambda,
                                 const double dt, const double volume, const double mass, const Eigen::Matrix3d &C)
        -> Eigen::MatrixXd {
    const double J = F.determinant();
}

// MPM Operations
auto nclr_p2g(const double inv_dx, const double hardening, const double mu, const double lambda, const double mass,
              const double dx, const double dt, const double volume, Eigen::Tensor<double, 4> &grid_velocity,
              Eigen::Tensor<double, 4> &grid_mass, std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v,
              std::vector<Eigen::Matrix3d> &F, std::vector<Eigen::Matrix3d> &C, Eigen::VectorXi, int model) -> void;

auto nclr_g2p(const double inv_dx, const double dt, Eigen::Tensor<double, 4> &grid_velocity,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, Eigen::VectorXi, int model) -> void;

auto nclr_grid_op(const int grid_resolution, const double dx, const double dt, const double gravity,
                  Eigen::Tensor<double, 4> &grid_velocity, Eigen::Tensor<double, 4> &grid_mass) -> void {
    constexpr int boundary = 3;
    const double v_allowed = dx * 0.9 / dt;

#pragma omp parallel for collapse(3)
    for (int ii = 0; ii <= grid_resolution; ++ii) {
        for (int jj = 0; jj <= grid_resolution; ++jj) {
            for (int kk = 0; kk <= grid_resolution; ++kk) {
                if (grid_mass(ii, jj, kk, 0) > 0.0) {
                    grid_velocity(ii, jj, kk, 0) /= grid_mass(ii, jj, kk, 0);
                    grid_velocity(ii, jj, kk, 1) /= grid_mass(ii, jj, kk, 0);
                    grid_velocity(ii, jj, kk, 2) /= grid_mass(ii, jj, kk, 0);

                    grid_velocity(ii, jj, kk, 1) += dt * gravity;

                    grid_velocity(ii, jj, kk, 0) = clamp(grid_velocity(ii, jj, kk, 0), -v_allowed, v_allowed);
                    grid_velocity(ii, jj, kk, 1) = clamp(grid_velocity(ii, jj, kk, 1), -v_allowed, v_allowed);
                    grid_velocity(ii, jj, kk, 2) = clamp(grid_velocity(ii, jj, kk, 2), -v_allowed, v_allowed);
                }

                if (ii < boundary && grid_velocity(ii, jj, kk, 0) < 0) { grid_velocity(ii, jj, kk, 0) = 0; }
                if (ii >= grid_resolution - boundary && grid_velocity(ii, jj, kk, 0) > 0) {
                    grid_velocity(ii, jj, kk, 0) = 0;
                }

                if (jj < boundary && grid_velocity(ii, jj, kk, 0) < 0) { grid_velocity(ii, jj, kk, 0) = 0; }
                if (jj >= grid_resolution - boundary && grid_velocity(ii, jj, kk, 0) > 0) {
                    grid_velocity(ii, jj, kk, 0) = 0;
                }

                if (kk < boundary && grid_velocity(ii, jj, kk, 0) < 0) { grid_velocity(ii, jj, kk, 0) = 0; }
                if (kk >= grid_resolution - boundary && grid_velocity(ii, jj, kk, 0) > 0) {
                    grid_velocity(ii, jj, kk, 0) = 0;
                }
            }
        }
    }
}

auto nclr_mpm(const double inv_dx, const double hardening, const double mu_0, const double lambda_0, const double mass,
              const double dx, const double dt, const double volume, const double res, const std::size_t timesteps,
              const Eigen::MatrixX3d &x) -> Eigen::MatrixXd {
    std::vector<std::vector<double>> out;
    out.reserve(timesteps);

    for (int ii = 0; ii < timesteps; ++ii) {}
}

PYBIND11_MODULE(nuclear_mpm, m) {
    m.doc() = "Fast offline MPM solver";

    m.def("nclr_mpm", &nclr_mpm);
}
