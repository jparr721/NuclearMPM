/* #include "nclr.h" */
/* #include <cstdint> */
/* #include "tqdm.h" */
/* #include <Eigen/SVD> */
/* #include <algorithm> */
/* #include <cmath> */
/* #include <fstream> */

/* namespace py = pybind11; */
#include "nclr.h"
#include "taichi.h"
#include <Eigen/Dense>
#include <cstdint>

auto nclr_constant_hardening(const double mu, const double lambda, const double e) -> std::pair<double, double> {
    return std::make_pair(mu * e, lambda * e);
}

auto nclr_snow_hardening(const double mu, const double lambda, const double h, const double jp)
        -> std::pair<double, double> {
    const double e = std::exp(h * (1.0 - jp));
    return nclr_constant_hardening(mu, lambda, e);
}


/* // Cauchy stress */
/* auto nclr_fixed_corotated_stress(const Eigen::Matrix3d &F, const double inv_dx, const double mu, const double lambda, */
/*                                  const double dt, const double volume, const double mass, const Eigen::Matrix3d &C) */
/*         -> Eigen::MatrixXd { */
/*     const double J = F.determinant(); */
/*     const double D_inv = 4 * inv_dx * inv_dx; */

/*     const auto &[R, S] = nclr_polar(F); */

/*     const Eigen::Matrix3d corotation_component = (F - R) * F.transpose(); */
/*     const Eigen::MatrixXd PF = (2 * mu * corotation_component) + Eigen::Matrix3d::Constant(lambda * (J - 1) * J); */
/*     const Eigen::MatrixXd stress = -(dt * volume) * (D_inv * PF); */
/*     return stress + mass * C; */
/* } */

/* // MPM Operations */
/* auto nclr_p2g(const int res, const double inv_dx, const double hardening, const double mu_0, const double lambda_0, */
/*               const double mass, const double dx, const double dt, const double volume, */
/*               std::vector<Eigen::Vector3d> &grid_velocity, std::vector<double> &grid_mass, */
/*               std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F, */
/*               std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void { */
/*     for (int ii = 0; ii < x.size(); ++ii) { */
/*         const Eigen::Vector3i base_coord = ((x.at(ii) * inv_dx) - Eigen::Vector3d::Constant(0.5)).cast<int>(); */
/*         if (oob(base_coord, res)) { */
/*             std::cout << base_coord.transpose() << std::endl; */
/*             throw std::runtime_error("Out of bounds."); */
/*         } */
/*         const Eigen::Vector3d fx = x.at(ii) * inv_dx - base_coord.cast<double>(); */

/*         const Eigen::Vector3d w_i = sqr(Eigen::Vector3d::Constant(1.5) - fx) * 0.5; */
/*         const Eigen::Vector3d w_j = sqr(fx - Eigen::Vector3d::Constant(1.0)) - Eigen::Vector3d::Constant(0.75); */
/*         const Eigen::Vector3d w_k = sqr(fx - Eigen::Vector3d::Constant(0.5)) * 0.5; */

/*         const auto [mu, lambda] = model == MaterialModel::kNeoHookean */
/*                                           ? nclr_constant_hardening(mu_0, lambda_0, hardening) */
/*                                           : nclr_snow_hardening(mu_0, lambda_0, hardening, Jp.at(ii)); */
/*         const Eigen::Matrix3d affine = */
/*                 nclr_fixed_corotated_stress(F.at(ii), inv_dx, mu, lambda, dt, volume, mass, C.at(ii)); */
/*         for (int jj = 0; jj < 3; ++jj) { */
/*             for (int kk = 0; kk < 3; ++kk) { */
/*                 for (int ll = 0; ll < 3; ++ll) { */
/*                     if (oob(base_coord, res, Eigen::Vector3i(jj, kk, ll))) { */
/*                         throw std::runtime_error("Out of bounds."); */
/*                     } */
/*                     const Eigen::Vector3d dpos = dx * (Eigen::Vector3d(jj, kk, ll) - fx); */
/*                     const Eigen::Vector3d mv = v.at(ii) * mass; */
/*                     const double weight = w_i(0) * w_j(1) * w_k(2); */

/*                     const Eigen::Vector3d v_term = weight * (mv + affine * dpos); */
/*                     const double m_term = weight * mass; */
/*                     const int index = base_coord[0] + jj * res + base_coord[1] + kk * res + base_coord[2] + ll; */
/*                     grid_velocity.at(index) = weight * (mv + affine * dpos); */
/*                     grid_mass.at(index) = weight * mass; */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */

/* auto nclr_g2p(const int res, const double inv_dx, const double dt, const std::vector<Eigen::Vector3d> &grid_velocity, */
/*               std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector3d> &v, std::vector<Eigen::Matrix3d> &F, */
/*               std::vector<Eigen::Matrix3d> &C, std::vector<double> &Jp, MaterialModel model) -> void { */
/* #pragma omp parallel for */
/*     for (int ii = 0; ii < x.size(); ++ii) { */
/*         const Eigen::Vector3i base_coord = (x.at(ii) * inv_dx - Eigen::Vector3d::Constant(0.5)).cast<int>(); */
/*         if (oob(base_coord, res)) { throw std::runtime_error("Out of bounds."); } */
/*         const Eigen::Vector3d fx = x.at(ii) * inv_dx - base_coord.cast<double>(); */

/*         const Eigen::Vector3d w_i = sqr(Eigen::Vector3d::Constant(1.5) - fx) * 0.5; */
/*         const Eigen::Vector3d w_j = sqr(fx - Eigen::Vector3d::Constant(1.0)) - Eigen::Vector3d::Constant(0.75); */
/*         const Eigen::Vector3d w_k = sqr(fx - Eigen::Vector3d::Constant(0.5)) * 0.5; */

/*         C.at(ii) = Eigen::Matrix3d::Zero(); */
/*         v.at(ii) = Eigen::Vector3d::Zero(); */

/*         for (int jj = 0; jj < 3; ++jj) { */
/*             for (int kk = 0; kk < 3; ++kk) { */
/*                 for (int ll = 0; ll < 3; ++ll) { */
/*                     if (oob(base_coord, res, Eigen::Vector3i(jj, kk, ll))) { */
/*                         std::cout << base_coord.transpose() << std::endl; */
/*                         throw std::runtime_error("Out of bounds."); */
/*                     } */
/*                     const Eigen::Vector3d dpos = Eigen::Vector3d(jj, kk, ll) - fx; */

/*                     const int index = base_coord[0] + jj * res + base_coord[1] + kk * res + base_coord[2] + ll; */
/*                     const Eigen::Vector3d grid_v = grid_velocity.at(index); */

/*                     const double weight = w_i(0) * w_j(1) * w_k(2); */
/*                     v.at(ii) += weight * grid_v; */
/*                     C.at(ii) += 4 * inv_dx * ((weight * grid_v) * dpos.transpose()); */
/*                 } */
/*             } */
/*         } */

/*         x.at(ii) += dt * v.at(ii); */
/*         Eigen::Matrix3d F_ = (diagonal(1.0) + dt * C.at(ii)) * F.at(ii); */

/*         Eigen::Vector3d _sig; */
/*         Eigen::Matrix3d U, V; */
/*         nclr_svd(F_, U, _sig, V); */

/*         Eigen::Matrix3d sig; */
/*         if (model == MaterialModel::kSnow) { */
/*             sig(0, 0) = std::clamp(sig(0), 1.0 - 2.5e-2, 1.0 + 7.5e-3); */
/*             sig(1, 1) = std::clamp(sig(1), 1.0 - 2.5e-2, 1.0 + 7.5e-3); */
/*             sig(2, 2) = std::clamp(sig(2), 1.0 - 2.5e-2, 1.0 + 7.5e-3); */
/*         } */

/*         const double old_J = F_.determinant(); */

/*         if (model == MaterialModel::kSnow) { F_ = U * sig * V.transpose(); } */

/*         const double det = F_.determinant() + 1e-10; */
/*         Jp.at(ii) = std::clamp(Jp.at(ii) * old_J / det, 0.6, 20.0); */
/*         F.at(ii) = F_; */
/*     } */
/* } */

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

// Window
constexpr int window_size = 1200;

// Grid resolution (cells)
const int n = 80;

const real dt = 1e-4f;
const real frame_dt = 1e-3f;
const real dx = 1.0f / n;
const real inv_dx = 1.0f / dx;

// Snow material properties
const auto particle_mass = 1.0f;
const auto vol = 1.0f;      // Particle Volume
const auto hardening = 0.7f;// Snow hardening factor
const auto E = 1400;        // Young's Modulus
const auto nu = 0.3f;       // Poisson ratio
const bool plastic = true;
const auto gravity = -90.8f;
int step = 0;

// Initial Lamé parameters
const real mu_0 = E / (2 * (1 + nu));
const real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

std::vector<Particle> particles;
int nn = n + 1;

auto nclr_grid_op(const float gravity, std::vector<Cell> &grid) -> void {
    // boundary thickness
    constexpr int boundary = 3;
#pragma omp parallel for collpase(2)
    for (int ii = 0; ii <= n; ii++) {
        for (int jj = 0; jj <= n; jj++) {
            const auto index = (ii * nn) + jj;
            auto &g = grid[index];
            // No need for epsilon here
            if (g.mass > 0.0) {
                // Normalize by mass
                g.velocity /= g.mass;
                // Gravity
                g.velocity(1) += dt * gravity;
            }

            // Node coordinates
            /* real x = real(i) / n; */
            /* real y = real(j) / n; */

            // Boundary conditions
            /* if (ii < boundary && grid.at(index).velocity(0) < 0) { grid.at(index).velocity(0) = 0; } */
            /* if (ii >= nn - boundary && grid.at(index).velocity(0) > 0) { grid.at(index).velocity(0) = 0; } */
            /* if (jj < boundary && grid.at(index).velocity(1) < 0) { grid.at(index).velocity(1) = 0; } */
            /* if (jj >= nn - boundary && grid.at(index).velocity(1) > 0) { grid.at(index).velocity(1) = 0; } */

            if (ii < boundary && grid.at(index).velocity(0) < 0 ||
                ii >= nn - boundary && grid.at(index).velocity(0) > 0 ||
                jj < boundary && grid.at(index).velocity(1) < 0 ||
                jj >= nn - boundary && grid.at(index).velocity(1) > 0) {
                g.velocity = constvec(0);
                g.mass = 0.0;
            }

            // Sticky boundary
            /* if (x < boundary || x > 1 - boundary || y > 1 - boundary || y < boundary) { */
            /*     g.velocity = constvec(0); */
            /*     g.mass = 0.0; */
            /* } */
            // Separate boundary (needs friction)
            /* if (y < boundary) { g.velocity[1] = std::max(0.0f, g.velocity[1]); } */
        }
    }
}

void advance(real dt) {
    auto grid = std::vector<Cell>(nn * nn, Cell());

    // P2G
    for (auto &p : particles) {
        // element-wise floor
        const Eigen::Vector2i base_coord = (p.x * inv_dx - constvec(0.5f)).cast<int>();

        const Vector<real> fx = p.x * inv_dx - base_coord.cast<real>();

        // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
        std::vector<Vector<real>> w{constvec(0.5).cwiseProduct(Eigen::square((constvec(1.5) - fx).array()).matrix()),
                                    constvec(0.75) - Eigen::square((fx - constvec(1.0)).array()).matrix(),
                                    constvec(0.5).cwiseProduct(Eigen::square((fx - constvec(0.5)).array()).matrix())};

        // Compute current Lamé parameters [http://mpm.graphics Eqn. 86]
        const auto mu = mu_0 * hardening;
        const auto lambda = lambda_0 * hardening;

        // Current volume
        const real J = p.F.determinant();

        // Polar decomposition for fixed corotated model
        Matrix<real> r, s;
        nclr_polar(p.F, r, s);
        /* polar_decomp(p.F, r, s); */

        // [http://mpm.graphics Paragraph after Eqn. 176]
        const real Dinv = 4 * inv_dx * inv_dx;

        // [http://mpm.graphics Eqn. 52]
        const Matrix<real> PF = (2 * mu * (p.F - r) * p.F.transpose() + constmat(lambda * (J - 1) * J));

        // Cauchy stress times dt and inv_dx
        const Matrix<real> stress = -(dt * vol) * (Dinv * PF);

        // Fused APIC momentum + MLS-MPM stress contribution
        // See http://taichi.graphics/wp-content/uploads/2019/03/mls-mpm-cpic.pdf
        // Eqn 29
        const Matrix<real> affine = stress + particle_mass * p.C;

        // P2G
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const Vector<real> dpos = (Vector<real>(i, j) - fx) * dx;
                // Translational momentum
                const Vector<real> mass_x_velocity(p.v * particle_mass);
                const auto weight = w[i][0] * w[j][1];
                const auto index = ((base_coord.x() + i) * nn) + (base_coord.y() + j);
                grid[index].velocity += (weight * (mass_x_velocity + (affine * dpos)));
                grid[index].mass += weight * particle_mass;
            }
        }
    }

    nclr_grid_op(gravity, grid);

    // G2P
    for (auto &p : particles) {
        // element-wise floor
        const Vector<int> base_coord = (p.x * inv_dx - constvec(0.5f)).cast<int>();

        const Vector<real> fx = p.x * inv_dx - base_coord.cast<real>();

        // Quadratic kernels [http://mpm.graphics Eqn. 123, with x=fx, fx-1,fx-2]
        std::vector<Vector<real>> w{constvec(0.5).cwiseProduct(Eigen::square((constvec(1.5) - fx).array()).matrix()),
                                    constvec(0.75) - Eigen::square((fx - constvec(1.0)).array()).matrix(),
                                    constvec(0.5).cwiseProduct(Eigen::square((fx - constvec(0.5)).array()).matrix())};

        p.C = constmat(0);
        p.v = constvec(0);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const auto index = ((base_coord.x() + i) * nn) + (base_coord.y() + j);
                const Vector<real> dpos = (Vector<real>(i, j) - fx);
                const Vector<real> &grid_v = grid[index].velocity;
                const auto weight = w[i][0] * w[j][1];

                // Velocity
                p.v += weight * grid_v;

                // APIC C
                p.C += 4 * inv_dx * (weight * grid_v) * dpos.transpose();
            }
        }

        // Advection
        p.x += dt * p.v;

        // MLS-MPM F-update
        p.F = (diag(1) + dt * p.C) * p.F;
    }
}

// Seed particles with position and color
void add_object(const Vector<real> &center, int c) {
    // Randomly sample 1000 particles in the square
    for (int i = 0; i < 1000; i++) {
        particles.push_back(Particle((randvec() * 2.0f - constvec(1)) * 0.08f + center, c));
    }
}

int main() {
    taichi::GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
    auto &canvas = gui.get_canvas();

    add_object(Vector<real>(0.35, 0.35), 0xED553B);
    add_object(Vector<real>(0.55, 0.15), 0xED553B);
    add_object(Vector<real>(0.55, 0.85), 0xED553B);

    int frame = 0;

    // Main Loop
    for (step = 0;; step++) {
        // Advance simulation
        advance(dt);

        // Visualize frame
        if (step % int(frame_dt / dt) == 0) {
            // Clear background
            canvas.clear(0x112F41);
            // Box
            canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
            // Particles
            for (auto p : particles) { canvas.circle(taichi::Vector2(p.x)).radius(2).color(p.c); }
            // Update image
            gui.update();
        }
    }
}
