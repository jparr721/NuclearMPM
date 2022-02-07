#include "nclr.h"
#include "taichi.h"
#include <Eigen/Dense>
#include <cstdint>
#include <memory>

namespace nclr {
    auto nclr_constant_hardening(const float mu, const float lambda, const float e) -> std::pair<float, float> {
        return std::make_pair(mu * e, lambda * e);
    }

    auto nclr_snow_hardening(const float mu, const float lambda, const float h, const float jp)
            -> std::pair<float, float> {
        const float e = std::exp(h * (1.0 - jp));
        return nclr_constant_hardening(mu, lambda, e);
    }

    // Window
    constexpr int window_size = 800;

    // Grid resolution (cells)
    const int n = 80;

    const real dt = 1e-4f;
    const real frame_dt = 1e-3f;

    // Snow material properties
    int step = 0;
}// namespace nclr

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

    add_object(Vector<real>(0.35, 0.35), 0xED553B);
    add_object(Vector<real>(0.55, 0.15), 0xED553B);
    add_object(Vector<real>(0.55, 0.85), 0xED553B);

    auto sim = std::make_unique<MPMSimulation<2>>(particles);

    int frame = 0;

    // Main Loop
    for (step = 0;; step++) {
        // Advance simulation
        sim->advance();

        // Visualize frame
        if (step % int(frame_dt / dt) == 0) {
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
