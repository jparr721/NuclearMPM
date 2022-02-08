# NuclearMPM
NuclearMPM is a high-efficiency MPM implementation using CPU-bound parallelism with a focus on being as ebeddable as possible. This library contains no UI code or baked-in GUI and instead relies on the user wrapping it however they'd like.

All of the sources are contained in two header files: `nclr.h` and `nclr_math.h`. Any other headers are to run the example code in `nclr.cpp`

## Example Project
```cpp
#include "nclr.h"
#include "taichi.h" // Packaged in the repo, not required for embedding.
#include <cstdint>
#include <memory>

// How many dimensions to run the sim in (2 or 3)
constexpr int kDimension = 2;

// The resolution of the cube meshes.
constexpr int kResolution = 50;

// Window
constexpr int kWindowSize = 800;

// Timestep size
constexpr nclr::real dt = 1e-4f;

// Frame draw interval
constexpr nclr::real frame_dt = 1e-3f;

// The color to paint the points
constexpr int color = 0xED553B;

int main() {
    using namespace nclr;
    taichi::GUI gui("Nuclear MPM", kWindowSize, kWindowSize);
    auto &canvas = gui.get_canvas();

    std::vector<Particle<kDimension>> particles;

    // Draw a cube in section 0.4 to 0.6 in x->y
    auto cube_particles = cube<kResolution, kDimension>(0.4, 0.6);
    for (const auto &pos : cube_particles) { particles.emplace_back(Particle<kDimension>(pos, color)); }

    // Draw a cube in section 0.4 to 0.6 in x->y
    cube_particles = cube<kResolution, kDimension>(0.4, 0.6);

    // Move the cube to the bottom of the screen so it doesn't bounce
    for (auto &pos : cube_particles) {
        pos(1) -= 0.39;
        particles.emplace_back(Particle<2>(pos, color));
    }

    // Allocate a mutable simulation object
    auto sim = std::make_unique<MPMSimulation<kDimension>>(particles);

    // Main Loop
    for (uint64_t frame = 0;; ++frame) {

        // Advance simulation
        sim->advance();

        // Visualize frame
        if (frame % int(frame_dt / dt) == 0) {

            // Clear background
            canvas.clear(0x112F41);

            // Boundary Condition Box
            canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();

            if constexpr (kDimension == 3) {
                for (const auto &p : sim->particles()) {
                    // Scale the values coming out of the transform to 0-1 (your mileage _will_ vary)
                    const Vector<real, 2> pt = pt_3d_to_2d(p.x) / 4;

                    // Convert the point to a taichi primitive.
                    const auto point = taichi::Vector2(pt);

                    // Draw this circle
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
```

## Working on this project
### Requirements
You can install the necessary dependencies (on ubuntu/pop-os) with:
```bash
$ sudo apt update && sudo apt install -y libeigen3-dev libx11-dev # ninja-build if you want to use ninja
```
### Building
This project doesn't bark at you for doing an in-source build, but I _highly_ recommend against it. In the event that you want to run the sample project, just do the following:
```bash
$ mkdir build && cd build && cmake -GNinja .. && ninja
```

### Running
Once you've compiled, you can run this project as `./nuclear_mpm`
