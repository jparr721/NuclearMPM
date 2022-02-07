#include "nclr.h"
#include "taichi.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>

namespace nclr {
    constexpr int kDimension = 2;
    constexpr int kResolution = 50;
    // Window
    constexpr int kWindowSize = 800;

    const real dt = 1e-4f;
    const real frame_dt = 1e-3f;
}// namespace nclr

int main() {
    using namespace nclr;

    std::cout << "This is just an example project!" << std::endl;

    taichi::GUI gui("Nuclear MPM", kWindowSize, kWindowSize);
    auto &canvas = gui.get_canvas();

    std::vector<Particle<kDimension>> particles;
    auto add_object = [&particles](const Vector<real, kDimension> &center, int c) {
        for (int i = 0; i < 1000; i++) {
            particles.emplace_back(
                    Particle<kDimension>((randvec<kDimension>() * 2.0 - constvec<kDimension>(1)) * 0.08 + center, c));
        }
    };

    int c = 0xED553B;
    auto cube_particles = cube<kResolution, kDimension>(0.4, 0.6);
    for (const auto &pos : cube_particles) { particles.emplace_back(Particle<kDimension>(pos, c)); }
    cube_particles = cube<kResolution, kDimension>(0.4, 0.6);
    for (auto &pos : cube_particles) {
        pos(1) -= 0.39;
        particles.emplace_back(Particle<2>(pos, c));
    }

    auto sim = std::make_unique<MPMSimulation<kDimension>>(particles);

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

            if constexpr (kDimension == 3) {
                for (const auto &p : sim->particles()) {
                    const Vector<real, 2> pt = pt_3d_to_2d(p.x) / 4;
                    auto point = taichi::Vector2(pt);
                    canvas.circle(point).radius(2).color(p.c);
                }
            } else {
                // Particles
                for (const auto &p : sim->particles()) { canvas.circle(taichi::Vector2(p.x)).radius(2).color(p.c); }
            }

            // Update image
            gui.update();
        }
    }
}
