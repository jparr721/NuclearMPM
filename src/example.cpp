#include "nclr.h"
#include "visualizer.h"

#include <cstdint>
#include <memory>
#include <vector>

// How many dimensions to run the sim in (2 or 3)
#ifndef NCLR_EXAMPLE_DIMENSION
#define NCLR_EXAMPLE_DIMENSION 2
#endif
constexpr int kDimension = NCLR_EXAMPLE_DIMENSION;

// The resolution of the cube meshes.
constexpr int kResolution = 50;

// Window
constexpr int kWindowSize = 800;

// Timestep size
constexpr nclr::real dt = 1e-4f;

// Frame draw interval
constexpr nclr::real frame_dt = 1e-3f;

// The color to paint the points
constexpr int kColor = 0xED553B;

template<int dimension>
auto to_points(const std::vector<nclr::Particle<dimension>> &particles) -> std::vector<nclr::viz::Point> {
    auto points = std::vector<nclr::viz::Point>{};
    points.reserve(particles.size());
    for (const auto &particle : particles) {
        if constexpr (dimension == 3) {
            const nclr::Vector<nclr::real, 2> point = nclr::pt_3d_to_2d(particle.x) / 4;
            points.push_back({point(0), point(1), static_cast<std::uint32_t>(particle.c)});
        } else {
            points.push_back({particle.x(0), particle.x(1), static_cast<std::uint32_t>(particle.c)});
        }
    }
    return points;
}

int main() {
    using namespace nclr;
    viz::Visualizer visualizer("Nuclear MPM", kWindowSize, kWindowSize);

    std::vector<Particle<kDimension>> particles;

    // Draw a cube in section 0.4 to 0.6 in x->y
    auto cube_particles = cube<kDimension>(kResolution, 0.4, 0.6);
    for (const auto &pos : cube_particles) { particles.emplace_back(Particle<kDimension>(pos, kColor)); }

    // Draw a cube in section 0.4 to 0.6 in x->y
    cube_particles = cube<kDimension>(kResolution, 0.4, 0.6);

    // Move the cube to the bottom of the screen so it doesn't bounce
    for (auto &pos : cube_particles) {
        pos(1) -= 0.35;
        particles.emplace_back(Particle<kDimension>(pos, kColor));
    }

    // Allocate a mutable simulation object, uncomment each for surprise!
    /* auto sim = std::make_unique<MPMSimulation<kDimension>>(particles, MaterialModel::kLiquid); */
    auto sim = std::make_unique<MPMSimulation<kDimension>>(particles, MaterialModel::kSnow);
    /* auto sim = std::make_unique<MPMSimulation<kDimension>>(particles, MaterialModel::kJelly); */

    for (std::uint64_t frame = 0; !visualizer.should_close(); ++frame) {
        sim->advance();

        if (frame % static_cast<int>(frame_dt / dt) != 0) { continue; }

        visualizer.draw(to_points(sim->particles()));
    }
}
