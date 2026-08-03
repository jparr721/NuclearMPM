# NuclearMPM

NuclearMPM is a high-efficiency material point method implementation using CPU-bound parallelism. The simulation is header-based (`src/nclr.h` and `src/nclr_math.h`) so it can be embedded without the example renderer.

The repository also includes:

- `nuclear_mpm_example`: an interactive OpenGL 4.1 particle visualization.
- `nuclear_mpm_solver`: a headless solver for generating simulation data.
- `src/visualizer.h`: the small GLFW/OpenGL interface used by the examples.

## Requirements

- A C++17 compiler
- CMake 3.16 or newer
- Ninja
- Git
- OpenGL development libraries
- Just (optional; used by the recommended `just` workflow)

CMake fetches pinned Eigen 3.4.0 and GLFW 3.4 sources automatically. The generated GLAD 2.0.8 OpenGL 4.1 loader is stored in `vendor/glad`, so builds do not require Python or a GLAD generator.

On Ubuntu or Pop!_OS, the system dependencies can be installed with:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libgl1-mesa-dev xorg-dev
```

Install Just from your distribution package or with `cargo install just`. The direct CMake workflow below does not require Just.

On macOS, install the build tools with Homebrew; OpenGL is provided by the system framework:

```bash
brew install cmake ninja just
```

## Building and Running with Just

```bash
just build
just run
```

`just build` initializes the `flags` submodule, creates `build/`, configures with `cmake -GNinja ..`, and runs Ninja. `just run` builds and launches `build/nuclear_mpm_example`.

Run the compile and solver smoke tests with:

```bash
ctest --test-dir build --output-on-failure
```

Remove generated build output with:

```bash
just clean
```

## Building Directly

```bash
mkdir -p build
cd build
cmake -GNinja ..
ninja
```

Enable assertions and optional visualization playback in the solver with:

```bash
cmake -GNinja -DWITH_NCLR_DEBUG=ON -DWITH_NCLR_SOLVER_VIZ=ON ..
ninja
```

## OpenGL Visualizer

The example renderer requests an OpenGL 4.1 Core context and uses GLSL 410 vertex and fragment shaders. Particle positions are uploaded through a dynamic VBO and rendered as points; the simulation boundary is rendered as a line loop. Close the window or press Escape to stop the example.

Simulation code only passes renderer-neutral values:

```cpp
#include "visualizer.h"

nclr::viz::Visualizer visualizer("Nuclear MPM", 800, 800);
std::vector<nclr::viz::Point> points{
        {0.5F, 0.5F, 0xED553B},
};

while (!visualizer.should_close()) {
    visualizer.draw(points);
}
```

## Headless Solver

Show the available options:

```bash
./build/nuclear_mpm_solver --help
```

Run a sample simulation and dump its states:

```bash
./build/nuclear_mpm_solver \
  --dump \
  --steps 4000 \
  --cube0-x 0.4 \
  --cube0-y 0.6
```

The solver does not overwrite generated datasets, so remove or move the existing output before generating another dataset.

## Embedding the Simulation

Include `src/nclr.h` in a C++17 target and link Eigen. The simulation headers do not depend on GLFW, GLAD, or OpenGL; those dependencies are isolated to `src/visualizer.cpp` and the example targets.

The material point method can be sensitive to grid indexing, particularly with the snow model. Start from `src/example.cpp`, verify that a simulation converges, and then move larger data-generation workloads to `nuclear_mpm_solver`.
