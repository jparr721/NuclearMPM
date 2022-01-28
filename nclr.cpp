#include "nclr.h"
#include "tqdm.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <fstream>

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
    const double D_inv = 4 * inv_dx * inv_dx;

    const auto &[R, S] = nclr_polar(F);

    const Eigen::Matrix3d corotation_component = (F - R) * F.transpose();
    const Eigen::MatrixXd PF = (2 * mu * corotation_component) + Eigen::Matrix3d::Constant(lambda * (J - 1) * J);
    const Eigen::MatrixXd stress = -(dt * volume) * (D_inv * PF);
    return stress + mass * C;
}

// MPM Operations
auto nclr_p2g(const int res, const double inv_dx, const double hardening, const double mu_0, const double lambda_0,
              const double mass, const double dx, const double dt, const double volume,
              std::vector<Eigen::Vector3d> &grid_velocity, std::vector<double> &grid_mass,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void {
    for (int ii = 0; ii < x.size(); ++ii) {
        const Eigen::Vector3i base_coord = ((x.at(ii) * inv_dx) - Eigen::Vector3d::Constant(0.5)).cast<int>();
        if (oob(base_coord, res)) {
            std::cout << base_coord.transpose() << std::endl;
            throw std::runtime_error("Out of bounds.");
        }
        const Eigen::Vector3d fx = x.at(ii) * inv_dx - base_coord.cast<double>();

        const Eigen::Vector3d w_i = sqr(Eigen::Vector3d::Constant(1.5) - fx) * 0.5;
        const Eigen::Vector3d w_j = sqr(fx - Eigen::Vector3d::Constant(1.0)) - Eigen::Vector3d::Constant(0.75);
        const Eigen::Vector3d w_k = sqr(fx - Eigen::Vector3d::Constant(0.5)) * 0.5;

        const auto [mu, lambda] = model == MaterialModel::kNeoHookean
                                          ? nclr_constant_hardening(mu_0, lambda_0, hardening)
                                          : nclr_snow_hardening(mu_0, lambda_0, hardening, Jp.at(ii));
        const Eigen::Matrix3d affine =
                nclr_fixed_corotated_stress(F.at(ii), inv_dx, mu, lambda, dt, volume, mass, C.at(ii));
        for (int jj = 0; jj < 3; ++jj) {
            for (int kk = 0; kk < 3; ++kk) {
                for (int ll = 0; ll < 3; ++ll) {
                    if (oob(base_coord, res, Eigen::Vector3i(jj, kk, ll))) {
                        throw std::runtime_error("Out of bounds.");
                    }
                    const Eigen::Vector3d dpos = dx * (Eigen::Vector3d(jj, kk, ll) - fx);
                    const Eigen::Vector3d mv = v.at(ii) * mass;
                    const double weight = w_i(0) * w_j(1) * w_k(2);

                    const Eigen::Vector3d v_term = weight * (mv + affine * dpos);
                    const double m_term = weight * mass;
                    const int index = base_coord[0] + jj * res + base_coord[1] + kk * res + base_coord[2] + ll;
                    grid_velocity.at(index) = weight * (mv + affine * dpos);
                    grid_mass.at(index) = weight * mass;
                }
            }
        }
    }
}

auto nclr_g2p(const int res, const double inv_dx, const double dt, const std::vector<Eigen::Vector3d> &grid_velocity,
              std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F,
              std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void {
#pragma omp parallel for
    for (int ii = 0; ii < x.size(); ++ii) {
        const Eigen::Vector3i base_coord = (x.at(ii) * inv_dx - Eigen::Vector3d::Constant(0.5)).cast<int>();
        if (oob(base_coord, res)) { throw std::runtime_error("Out of bounds."); }
        const Eigen::Vector3d fx = x.at(ii) * inv_dx - base_coord.cast<double>();

        const Eigen::Vector3d w_i = sqr(Eigen::Vector3d::Constant(1.5) - fx) * 0.5;
        const Eigen::Vector3d w_j = sqr(fx - Eigen::Vector3d::Constant(1.0)) - Eigen::Vector3d::Constant(0.75);
        const Eigen::Vector3d w_k = sqr(fx - Eigen::Vector3d::Constant(0.5)) * 0.5;

        C.at(ii) = Eigen::Matrix3d::Zero();
        v.at(ii) = Eigen::Vector3d::Zero();

        for (int jj = 0; jj < 3; ++jj) {
            for (int kk = 0; kk < 3; ++kk) {
                for (int ll = 0; ll < 3; ++ll) {
                    if (oob(base_coord, res, Eigen::Vector3i(jj, kk, ll))) {
                        std::cout << base_coord.transpose() << std::endl;
                        throw std::runtime_error("Out of bounds.");
                    }
                    const Eigen::Vector3d dpos = Eigen::Vector3d(jj, kk, ll) - fx;

                    const int index = base_coord[0] + jj * res + base_coord[1] + kk * res + base_coord[2] + ll;
                    const Eigen::Vector3d grid_v = grid_velocity.at(index);

                    const double weight = w_i(0) * w_j(1) * w_k(2);
                    v.at(ii) += weight * grid_v;
                    C.at(ii) += 4 * inv_dx * ((weight * grid_v) * dpos.transpose());
                }
            }
        }

        x.at(ii) += dt * v.at(ii);
        Eigen::Matrix3d F_ = (diagonal(1.0) + dt * C.at(ii)) * F.at(ii);

        Eigen::Vector3d _sig;
        Eigen::Matrix3d U, V;
        nclr_svd(F_, U, _sig, V);

        Eigen::Matrix3d sig;
        if (model == MaterialModel::kSnow) {
            sig(0, 0) = std::clamp(sig(0), 1.0 - 2.5e-2, 1.0 + 7.5e-3);
            sig(1, 1) = std::clamp(sig(1), 1.0 - 2.5e-2, 1.0 + 7.5e-3);
            sig(2, 2) = std::clamp(sig(2), 1.0 - 2.5e-2, 1.0 + 7.5e-3);
        }

        const double old_J = F_.determinant();

        if (model == MaterialModel::kSnow) { F_ = U * sig * V.transpose(); }

        const double det = F_.determinant() + 1e-10;
        Jp.at(ii) = std::clamp(Jp.at(ii) * old_J / det, 0.6, 20.0);
        F.at(ii) = F_;
    }
}

auto nclr_grid_op(const int res, const double dx, const double dt, const double gravity,
                  const std::vector<double> &grid_mass, std::vector<Eigen::Vector3d> &grid_velocity) -> void {
    constexpr int boundary = 3;
    const double v_allowed = dx * 0.9 / dt;

#pragma omp parallel for collapse(3)
    for (int ii = 0; ii <= res; ++ii) {
        for (int jj = 0; jj <= res; ++jj) {
            for (int kk = 0; kk <= res; ++kk) {

                // Grid normalization
                const int index = ii * res + jj * res + kk;
                if (grid_mass.at(index) > 0.0) {
                    grid_velocity.at(index) /= grid_mass[index];

                    grid_velocity.at(index)(1) += dt * gravity;
                    grid_velocity.at(index)(0) = std::clamp(grid_velocity.at(index)(0), -v_allowed, v_allowed);
                    grid_velocity.at(index)(1) = std::clamp(grid_velocity.at(index)(1), -v_allowed, v_allowed);
                    grid_velocity.at(index)(2) = std::clamp(grid_velocity.at(index)(2), -v_allowed, v_allowed);
                }

                // Boundary conditions
                if (ii < boundary && grid_velocity.at(index)(0) < 0) { grid_velocity.at(index)(0) = 0; }
                if (ii >= res - boundary && grid_velocity.at(index)(0) > 0) { grid_velocity.at(index)(0) = 0; }
                if (jj < boundary && grid_velocity.at(index)(1) < 0) { grid_velocity.at(index)(1) = 0; }
                if (jj >= res - boundary && grid_velocity.at(index)(1) > 0) { grid_velocity.at(index)(1) = 0; }
                if (kk < boundary && grid_velocity.at(index)(2) < 0) { grid_velocity.at(index)(2) = 0; }
                if (kk >= res - boundary && grid_velocity.at(index)(2) > 0) { grid_velocity.at(index)(2) = 0; }
            }
        }
    }
}

auto nclr_mpm(const double inv_dx, const double hardening, const double mu_0, const double lambda_0, const double mass,
              const double dx, const double dt, const double volume, const unsigned int res, const double gravity,
              const std::size_t timesteps, const Eigen::MatrixX3d &x) -> std::vector<Eigen::MatrixXd> {
    std::vector<std::vector<Eigen::Vector3d>> out;
    out.reserve(timesteps);

    std::vector<Eigen::Vector3d> positions;
    for (int row = 0; row < x.rows(); ++row) { positions.push_back(x.row(row)); }
    auto v = std::vector<Eigen::Vector3d>(x.rows(), Eigen::Vector3d::Zero());
    auto F = std::vector<Eigen::Matrix3d>(x.rows(), diagonal(1.0));
    auto C = std::vector<Eigen::Matrix3d>(x.rows(), Eigen::Matrix3d::Zero());
    auto Jp = std::vector<double>(x.rows(), 1);

    for (int ii : tqdm::range(timesteps)) {
        auto grid_velocity = std::vector<Eigen::Vector3d>(res * res * res, Eigen::Vector3d::Zero());
        auto grid_mass = std::vector<double>(res * res * res, 0);

        nclr_p2g(res, inv_dx, hardening, mu_0, lambda_0, mass, dx, dt, volume, grid_velocity, grid_mass, positions, v,
                 F, C, Jp, MaterialModel::kNeoHookean);
        nclr_grid_op(res, dx, dt, gravity, grid_mass, grid_velocity);
        nclr_g2p(res, inv_dx, dt, grid_velocity, positions, v, F, C, Jp, MaterialModel::kNeoHookean);
        out.push_back(positions);
    }

    std::vector<Eigen::MatrixXd> all_positions;

    for (const auto &pos : out) { all_positions.push_back(vec_to_mat(pos)); }

    return all_positions;
}

PYBIND11_MODULE(nuclear_mpm, m) {
    m.doc() = "Fast offline MPM solver";

    m.def("nclr_mpm", &nclr_mpm);
    m.def("nclr_fixed_corotated_stress", &nclr_fixed_corotated_stress);
    m.def("nclr_polar", &nclr_polar);
}
