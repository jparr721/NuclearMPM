#include "nclr.h"
#include "taichi.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>

namespace nclr {
    // Window
    constexpr int window_size = 800;

    const real dt = 1e-4f;
    const real frame_dt = 1e-3f;
}// namespace nclr

auto map_3d() -> void {}

int main() {
    using namespace nclr;
    taichi::GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
    auto &canvas = gui.get_canvas();

    std::vector<Particle<2>> particles;
    auto add_object = [&particles](const Vector<real, 2> &center, int c) {
        for (int i = 0; i < 1000; i++) {
            particles.emplace_back(Particle<2>((randvec<2>() * 2.0f - constvec<2>(1)) * 0.08f + center, c));
        }
    };

    /* add_object(Vector<real>(0.50, 0.10), 0xED553B); */
    int c = 0xED553B;
    auto cube_particles = cube<50>(0.4, 0.6, 0.6, 0.8);
    for (const auto &pos : cube_particles) { particles.emplace_back(Particle<2>(pos, c)); }
    /* cube_particles = cube<50>(0.4, 0.6, 0.01, 0.21); */
    /* for (const auto &pos : cube_particles) { particles.emplace_back(Particle<2>(pos, c)); } */

    /* Matrix<real, -1> GV; */
    /* Vector<real, 2> res(50, 50); */
    /* grid(res, GV); */
    /* GV /= 4; */
    /* GV.col(0) = (GV.col(0).array() + 0.3).matrix(); */
    /* GV.col(1) = (GV.col(1).array() + 0.1).matrix(); */

    /* const real k = 0.2; */
    /* const real t = 0.3; */
    /* for (auto row = 0; row < GV.rows(); ++row) { */
    /*     const Vector<real, 2> &pos = GV.row(row); */
    /*     const auto iso = gyroid<50>(k, t, pos); */
    /*     particles.emplace_back(Particle<2>(pos, c)); */
    /* } */

    auto sim = std::make_unique<MPMSimulation<2>>(particles);

    uint64_t frame = 0;

    // Main Loop
    for (;; ++frame) {
        // Advance simulation
        sim->advance();

        // Visualize frame
        if (frame % int(frame_dt / dt) == 0) {
            // Clear background
            canvas.clear(0x112F41);
            // Box
            canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
            // Particles
            for (const auto &p : sim->particles()) { canvas.circle(taichi::Vector2(p.x)).radius(2).color(p.c); }
            // Update image
            gui.update();
        }
    }
}
