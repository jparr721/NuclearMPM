#include "nclr_math.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

// Avoid conflicting declaration of min/max macros in windows headers
#if !defined(NOMINMAX) && (defined(_WIN32) || defined(_WIN32_) || defined(WIN32) || defined(_WIN64))
#define NOMINMAX
#ifdef max
#undef max
#undef min
#endif
#endif

namespace nclr {
    template<int dim>
    struct Particle {
        // Position
        Vector<real, dim> x;

        // Velocity
        Vector<real, dim> v;

        // Deformation gradient
        Matrix<real, dim> F;

        // Affine momentum from APIC
        Matrix<real, dim> C;

        // Determinant of the deformation gradient (i.e. volume)
        real Jp;

        // Mass
        real mass;

        // Volume (Per-Particle)
        real volume;

        // Color
        int c;

        Particle(Vector<real, dim> x, int c, Vector<real, dim> v = constvec<dim>(0), real mass = 1.0, real volume = 1.0)
            : x(x), v(v), F(diag<dim>(1)), C(constmat<dim>(0)), Jp(1.0), c(c), mass(mass), volume(volume) {}
    };

    template<int dim>
    struct Cell {
        Vector<real, dim> velocity;
        real mass;
        Cell() : velocity(constvec<dim>(0)), mass(0.0) {}
    };

    enum class MaterialModel {
        kSnow = 0,
        kJelly,
        kLiquid,
    };

    template<int dim>
    class MPMSimulation {
    public:
        constexpr static int kBoundary = 3;
        constexpr static nclr::real kSnowHardening = 10.0;
        constexpr static nclr::real kJellyHardening = 0.3;
        constexpr static nclr::real kLiquidHardening = 1.0;

        const real mu_0;
        const real lambda_0;

        MPMSimulation(std::vector<Particle<dim>> particles, const MaterialModel model, int res = 64, real dt = 1e-4,
                      real E = 1e4, real nu = 0.2, real gravity = -100)
            : particles_(std::move(particles)), material_model_(model), res_(res), dt_(dt), dx_(1.0 / res),
              inv_dx_(1 / dx_), E_(E), nu_(nu), gravity_(gravity), mu_0(E / (2 * (1 + nu))),
              lambda_0(E * nu / ((1 + nu) * (1 - 2 * nu))) {}

        auto advance() -> void {
            p2g();
            grid_op();
            g2p();
        }

        auto particles() const -> const std::vector<Particle<dim>> & { return particles_; }
        auto grid() const -> const std::vector<Cell<dim>> & { return cells_; }

    private:
        const MaterialModel material_model_;

        const int res_;

        const real dt_;
        const real dx_;
        const real inv_dx_;
        const real E_;
        const real nu_;
        const real gravity_;

        std::vector<Cell<dim>> cells_;
        std::vector<Particle<dim>> particles_;

        inline auto p2g() -> void {
            if constexpr (dim == 3) {
                cells_ = std::vector<Cell<dim>>((res_ + 1) * (res_ + 1) * (res_ + 1), Cell<dim>());
            } else {
                cells_ = std::vector<Cell<dim>>((res_ + 1) * (res_ + 1), Cell<dim>());
            }

#pragma omp parallel for
            for (auto pp = 0; pp < particles_.size(); ++pp) {
                auto &p = particles_.at(pp);
                // element-wise floor
                const Vector<int, dim> base_coord = (p.x * inv_dx_ - constvec<dim>(0.5)).template cast<int>();

#ifdef NCLR_DEBUG
                assert(!oob(base_coord));
#endif

                const Vector<real, dim> fx = p.x * inv_dx_ - base_coord.template cast<real>();

                // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
                std::vector<Vector<real, dim>> w{
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((constvec<dim>(1.5) - fx).array()).matrix()),
                        constvec<dim>(0.75) - Eigen::square((fx - constvec<dim>(1.0)).array()).matrix(),
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((fx - constvec<dim>(0.5)).array()).matrix())};

                const Matrix<real, dim> affine = first_piola_kirchoff_stress(p);

                // P2G
                for (int ii = 0; ii < 3; ++ii) {
                    for (int jj = 0; jj < 3; ++jj) {
                        if constexpr (dim == 3) {
#ifdef NCLR_DEBUG
                            assert(!oob(base_coord, Vector<real, dim>(ii, jj, kk)));
#endif
                            for (int kk = 0; kk < 3; ++kk) {
                                const Vector<real, dim> dpos = (Vector<real, dim>(ii, jj, kk) - fx) * dx_;
                                const auto weight = w[ii][0] * w[jj][1] * w[kk][2];
                                const auto index = ((base_coord.x() + ii) * (res_ + 1) * (res_ + 1)) +
                                                   ((base_coord.y() + jj) * (res_ + 1)) + (base_coord.z() + kk);
                                compute_fused_momentum(index, weight, dpos, affine, p);
                            }

                        } else {
#ifdef NCLR_DEBUG
                            assert(!oob(base_coord, Vector<real, dim>(ii, jj)));
#endif
                            const Vector<real, dim> dpos = (Vector<real, dim>(ii, jj) - fx) * dx_;
                            const auto weight = w[ii][0] * w[jj][1];
                            const auto index = ((base_coord.x() + ii) * (res_ + 1)) + (base_coord.y() + jj);
                            compute_fused_momentum(index, weight, dpos, affine, p);
                        }
                    }
                }
            }
        }

        inline auto compute_fused_momentum(const int index, const float weight, const Vector<real, dim> &dpos,
                                           const Matrix<real, dim> &affine, const Particle<dim> &particle) -> void {
            const Vector<real, dim> mass_x_velocity = particle.v * particle.mass;
            cells_.at(index).velocity += (weight * (mass_x_velocity + (affine * dpos)));
            cells_.at(index).mass += weight * particle.mass;
        }

        inline auto g2p() -> void {
#pragma omp parallel for
            for (auto pp = 0; pp < particles_.size(); ++pp) {
                auto &p = particles_.at(pp);
                // element-wise floor
                const Vector<int, dim> base_coord = (p.x * inv_dx_ - constvec<dim>(0.5)).template cast<int>();
#ifdef NCLR_DEBUG
                assert(!oob(base_coord));
#endif

                const Vector<real, dim> fx = p.x * inv_dx_ - base_coord.template cast<real>();

                // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
                std::vector<Vector<real, dim>> w{
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((constvec<dim>(1.5) - fx).array()).matrix()),
                        constvec<dim>(0.75) - Eigen::square((fx - constvec<dim>(1.0)).array()).matrix(),
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((fx - constvec<dim>(0.5)).array()).matrix())};

                p.C = constmat<dim>(0);
                p.v = constvec<dim>(0);

                for (int ii = 0; ii < 3; ++ii) {
                    for (int jj = 0; jj < 3; ++jj) {
                        if constexpr (dim == 3) {
                            for (int kk = 0; kk < 3; ++kk) {
#ifdef NCLR_DEBUG
                                assert(!oob(base_coord, Vector<real, dim>(ii, jj, kk)));
#endif
                                const Vector<real, dim> dpos = (Vector<real, dim>(ii, jj, kk) - fx);

                                const auto index = ((base_coord.x() + ii) * (res_ + 1) * (res_ + 1)) +
                                                   ((base_coord.y() + jj) * (res_ + 1)) + (base_coord.z() + kk);
                                const Vector<real, dim> &grid_v = cells_.at(index).velocity;
                                const auto weight = w[ii][0] * w[jj][1] * w[kk][2];

                                // Velocity
                                p.v += weight * grid_v;

                                // APIC C
                                p.C += 4 * inv_dx_ * (weight * grid_v) * dpos.transpose();
                            }

                        } else {
#ifdef NCLR_DEBUG
                            assert(!oob(base_coord, Vector<real, dim>(ii, jj)));
#endif
                            const Vector<real, dim> dpos = (Vector<real, dim>(ii, jj) - fx);

                            const auto index = ((base_coord.x() + ii) * (res_ + 1)) + (base_coord.y() + jj);
                            const Vector<real, dim> &grid_v = cells_.at(index).velocity;
                            const auto weight = w[ii][0] * w[jj][1];

                            // Velocity
                            p.v += weight * grid_v;

                            // APIC C
                            p.C += 4 * inv_dx_ * (weight * grid_v) * dpos.transpose();
                        }
                    }
                }

                // Advection
                p.x += dt_ * p.v;
                Matrix<real, dim> _F = (diag<dim>(1) + dt_ * p.C) * p.F;

                if (material_model_ == MaterialModel::kJelly) {
                    // MLS-MPM F-update for non-compressive elastic materials
                    p.F = _F;
                } else {
                    Matrix<real, dim> U, sig, V;
                    nclr_svd(_F, U, sig, V);

                    if (material_model_ == MaterialModel::kSnow) {
                        // Plasticity operation on sigma
#pragma unroll
                        for (int dd = 0; dd < dim; ++dd) {
                            sig(dd, dd) = std::clamp(sig(dd, dd), real(1.0 - 2.5e-2), real(1.0 + 4.5e-3));
                        }

                        const auto old_J = _F.determinant();
                        _F = U * sig * V.transpose();
                        p.Jp = std::clamp(p.Jp * old_J / _F.determinant(), real(0.6), real(20.0));
                        p.F = _F;
                    }

                    if (material_model_ == MaterialModel::kLiquid) {
                        auto J = 1.0;
                        for (int dd = 0; dd < dim; ++dd) { J *= sig(dd, dd); }
                        // Reset the deformation gradient to avoid numerical explosion
                        p.F = diag<dim>(1.0);
                        p.F(0, 0) = J;
                    }
                }
            }
        }

        inline auto grid_op() -> void {
#pragma omp parallel for collpase(dim)
            for (auto ii = 0; ii <= res_; ++ii) {
                for (auto jj = 0; jj <= res_; ++jj) {
                    if constexpr (dim == 3) {
                        for (auto kk = 0; kk <= res_; ++kk) {
                            const auto index = (ii * (res_ + 1) * (res_ + 1)) + (jj * (res_ + 1)) + kk;
                            auto &g = cells_.at(index);
                            grid_normalization(g);
                            sticky_boundary(Vector<real, dim>(ii, jj, kk), g);
                        }
                    } else {
                        const auto index = (ii * (res_ + 1)) + jj;
                        auto &g = cells_.at(index);
                        grid_normalization(g);
                        sticky_boundary(Vector<real, dim>(ii, jj), g);
                    }
                }
            }
        }

        inline auto grid_normalization(Cell<dim> &cell) -> void {
            const real allowed_velocity = dx_ * 0.9 / dt_;
            // No need for epsilon here
            if (cell.mass > 0.0) {
                // Normalize by mass
                cell.velocity /= cell.mass;

                // Apply Gravity to Y axis
                cell.velocity(1) += dt_ * gravity_;

                // Clip the grid velocity
                for (int dd = 0; dd < dim; ++dd) {
                    cell.velocity(dd) = std::clamp(cell.velocity(dd), -allowed_velocity, allowed_velocity);
                }
            }
        }

        inline auto sticky_boundary(const Vector<real, dim> &indices, Cell<dim> &cell) -> void {
#pragma unroll
            for (int dd = 0; dd < dim; ++dd) {
                if (indices(dd) < kBoundary && cell.velocity(dd) < 0 ||
                    indices(dd) >= (res_ + 1) - kBoundary && cell.velocity(dd) > 0) {
                    cell.velocity = constvec<dim>(0);
                    cell.mass = 0.0;
                }
            }
        }

        // Utilities ==============================================
        inline auto first_piola_kirchoff_stress(const Particle<dim> &p) -> Matrix<real, dim> {
            // Compute current Lamé parameters [http://mpm.graphics Eqn. 86] (for snow)
            const auto &[mu, lambda] = hardening(p);

            // Current volume
            const real J = p.F.determinant();

            // Polar decomposition for fixed corotated model
            Matrix<real, dim> r, s;
            nclr_polar(p.F, r, s);

            // [http://mpm.graphics Paragraph after Eqn. 176]
            const real Dinv = 4 * inv_dx_ * inv_dx_;

            // [http://mpm.graphics Eqn. 52]
            const Matrix<real, dim> PF = (2 * mu * (p.F - r) * p.F.transpose() + constmat<dim>(lambda * (J - 1) * J));

            // Cauchy stress times dt and inv_dx
            const Matrix<real, dim> stress = -(dt_ * p.volume) * (Dinv * PF);

            // Fused APIC momentum + MLS-MPM stress contribution
            // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
            // Eqn 29
            return stress + p.mass * p.C;// Affine MLS-MPM Stress update
        }

        // TODO(@jparr721) - Implement neo-hookean stress model.

        /**
         * Compute the hardening of the plasticity model as
         * F^n+1 = F^n+1_E + F_n+1_P
         * Where each component is the elastic and plastic components of the hardening model.
         * This is simplified as:
         * mu(F_P) = mu_0 * e^epsilon(1 - J_p)
         * lambda(F_P) = lambda_0 * e^epsilon(1 - J_p)
         * J_p (volume) is provided by the particle, so we just compute the value of e and
         * multiply through in this implementation.
         */
        inline auto constant_hardening(const real e) -> std::pair<real, real> {
            return std::make_pair<real, real>(mu_0 * e, lambda_0 * e);
        }

        inline auto snow_hardening(const Particle<dim> &p) -> std::pair<real, real> {
            const auto e = std::exp(kSnowHardening * (1.0 - p.Jp));
            return constant_hardening(e);
        }

        inline auto hardening(const Particle<dim> &p) -> std::pair<real, real> {
            switch (material_model_) {
                case MaterialModel::kSnow:
                    return snow_hardening(p);
                case MaterialModel::kJelly:
                    return constant_hardening(kJellyHardening);
                case MaterialModel::kLiquid:
                    return constant_hardening(kLiquidHardening);
                default:
                    std::cerr << "How" << std::endl;
                    return std::pair(0.0, 0.0);
            }
        }

        inline auto oob(const Vector<int, dim> base, const Vector<int, dim> ijk = Vector<int, dim>::Zero()) -> bool {
            const Vector<int, dim> bijk = base + ijk;
            const Vector<int, dim> comp = Vector<int, dim>::Ones() * res_;

#pragma unroll
            for (int ii = 0; ii < dim; ++ii) {
                if (bijk(ii) >= comp(ii) || bijk(ii) < 0) { return true; }
            }
            return false;
        }
    };
}// namespace nclr
