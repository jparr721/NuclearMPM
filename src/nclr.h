#include "math.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <utility>
#include <vector>

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

        Particle(Vector<real> x, int c, Vector<real> v = constvec<dim>(0), real mass = 1.0, real volume = 1.0)
            : x(x), v(v), F(diag(1)), C(constmat<dim>(0)), Jp(1), c(c), mass(mass), volume(volume) {}
    };

    template<int dim>
    struct Cell {
        Vector<real> velocity;
        double mass;
        Cell() : velocity(constvec<dim>(0)), mass(0.0) {}
    };

    template<int dim>
    class MPMSimulation {
    public:
        MPMSimulation(std::vector<Particle<dim>> particles, int res = 80, real dt = 1e-4, real frame_dt = 1e-3,
                      real hardening = 0.7, real E = 1000, real nu = 0.3, real gravity = -100)
            : particles_(std::move(particles)), res_(res), dt_(dt), frame_dt_(frame_dt), dx_(1.0 / res),
              inv_dx_(1 / dx_), hardening_(hardening), E_(E), nu_(nu), gravity_(gravity), mu_0_(E / (2 * (1 + nu))),
              lambda_0_(E * nu / ((1 + nu) * (1 - 2 * nu))) {}

        auto advance() -> void {
            p2g();
            grid_op();
            g2p();
        }

        auto particles() const -> const std::vector<Particle<dim>> & { return particles_; }

    private:
        static constexpr int kBoundary = 3;

        const int res_;

        const real dt_;
        const real frame_dt_;
        const real dx_;
        const real inv_dx_;
        const real hardening_;
        const real E_;
        const real nu_;
        const real gravity_;
        const real mu_0_;
        const real lambda_0_;

        std::vector<Cell<dim>> cells_;
        std::vector<Particle<dim>> particles_;

        inline auto p2g() -> void {
            cells_ = std::vector<Cell<dim>>((res_ + 1) * (res_ + 1) * (dim == 3 ? (res_ + 1) : 1), Cell<dim>());

#pragma omp parallel for
            for (auto pp = 0; pp < particles_.size(); ++pp) {
                auto &p = particles_.at(pp);
                // element-wise floor
                const Vector<int, dim> base_coord = (p.x * inv_dx_ - constvec<dim>(0.5f)).template cast<int>();

                const Vector<real, dim> fx = p.x * inv_dx_ - base_coord.template cast<real>();

                // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
                std::vector<Vector<real>> w{
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((constvec<dim>(1.5) - fx).array()).matrix()),
                        constvec<dim>(0.75) - Eigen::square((fx - constvec<dim>(1.0)).array()).matrix(),
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((fx - constvec<dim>(0.5)).array()).matrix())};

                // Compute current Lamé parameters [http://mpm.graphics Eqn. 86] (for snow)
                const auto mu = mu_0_ * hardening_;
                const auto lambda = lambda_0_ * hardening_;

                // Current volume
                const real J = p.F.determinant();

                // Polar decomposition for fixed corotated model
                Matrix<real, dim> r, s;
                nclr_polar(p.F, r, s);

                // [http://mpm.graphics Paragraph after Eqn. 176]
                const real Dinv = 4 * inv_dx_ * inv_dx_;

                // [http://mpm.graphics Eqn. 52]
                const Matrix<real, dim> PF =
                        (2 * mu * (p.F - r) * p.F.transpose() + constmat<dim>(lambda * (J - 1) * J));

                // Cauchy stress times dt and inv_dx
                const Matrix<real, dim> stress = -(dt_ * p.volume) * (Dinv * PF);

                // Fused APIC momentum + MLS-MPM stress contribution
                // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
                // Eqn 29
                const Matrix<real, dim> affine = stress + p.mass * p.C;

                // P2G
                for (int ii = 0; ii < 3; ++ii) {
                    for (int jj = 0; jj < 3; ++jj) {
                        if constexpr (dim == 3) {
                            for (int kk = 0; kk < 3; ++kk) {
                                const Vector<real, dim> dpos = (Vector<real, dim>(ii, jj, kk) - fx) * dx_;
                                const auto weight = w[ii][0] * w[jj][1] * w[kk][2];
                                const auto index = ((base_coord.x() + ii) * (res_ + 1) * (res_ + 1)) +
                                                   ((base_coord.y() + jj) * (res_ + 1)) + (base_coord.z() + kk);
                                compute_fused_momentum(index, weight, dpos, affine, p);
                            }

                        } else {
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
                const Vector<int, dim> base_coord = (p.x * inv_dx_ - constvec<dim>(0.5f)).template cast<int>();

                const Vector<real, dim> fx = p.x * inv_dx_ - base_coord.template cast<real>();

                // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
                std::vector<Vector<real>> w{
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((constvec<dim>(1.5) - fx).array()).matrix()),
                        constvec<dim>(0.75) - Eigen::square((fx - constvec<dim>(1.0)).array()).matrix(),
                        constvec<dim>(0.5).cwiseProduct(Eigen::square((fx - constvec<dim>(0.5)).array()).matrix())};

                p.C = constmat<dim>(0);
                p.v = constvec<dim>(0);

                for (int ii = 0; ii < 3; ii++) {
                    for (int jj = 0; jj < 3; jj++) {
                        if constexpr (dim == 3) {
                            for (int kk = 0; kk < 3; ++kk) {
                                const Vector<real, dim> dpos = (Vector<real>(ii, jj) - fx);

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
                            const Vector<real, dim> dpos = (Vector<real>(ii, jj) - fx);

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

                // MLS-MPM F-update
                p.F = (diag(1) + dt_ * p.C) * p.F;
            }
        }

        inline auto grid_op() -> void {
#pragma omp parallel for collpase(dim)
            for (auto ii = 0; ii <= res_; ++ii) {
                for (auto jj = 0; jj <= res_; ++jj) {
                    if (dim == 3) {
                        for (auto kk = 0; kk <= res_; ++kk) {
                            const auto index = (ii * (res_ + 1) * (res_ + 1)) + (jj * (res_ + 1)) + kk;
                            auto &g = cells_.at(index);
                            grid_normalization(g);
                            sticky_boundary(Vector<real, dim>(ii, jj), g);
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
            // No need for epsilon here
            if (cell.mass > 0.0) {
                // Normalize by mass
                cell.velocity /= cell.mass;

                // Apply Gravity to Y axis
                cell.velocity(1) += dt_ * gravity_;
            }
        }

        inline auto sticky_boundary(const Vector<real, dim> &indices, Cell<dim> &cell) -> void {
#pragma unroll
            for (int ii = 0; ii < dim; ++ii) {
                if (indices(ii) < kBoundary && cell.velocity(ii) < 0 ||
                    indices(ii) >= (res_ + 1) - kBoundary && cell.velocity(ii) > 0) {
                    cell.velocity = constvec<dim>(0);
                    cell.mass = 0.0;
                }
            }
        }

        inline auto oob(const Vector<int> base, const int res, const Vector<int> ijk = Vector<int>::Zero()) -> bool {
            const Vector<int> bijk = base + ijk;
            const Vector<int> comp = Vector<int>::Ones() * res;

#pragma unroll
            for (int ii = 0; ii < dim; ++ii) {
                if (bijk(ii) >= comp(ii) || bijk(ii) < 0) { return true; }
            }
            return false;
        }
    };
}// namespace nclr
