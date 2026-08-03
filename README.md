# NuclearMPM

NuclearMPM is a correctness-first C++17 implementation of the Moving Least Squares Material Point Method (MLS-MPM) with APIC transfers. The core library supports real 2D and 3D simulations, fixed-corotated solids, elastoplastic snow, weakly compressible liquids, configurable boundaries, CFL-controlled substeps, diagnostics, and optional memory-bounded OpenMP transfers.

The repository also provides:

- `nuclear_mpm_solver`: a validated 2D/3D command-line solver with versioned snapshots.
- `nuclear_mpm_example`: a Dear ImGui simulation workbench with persistent 2D/3D sessions.
- A native OpenGL 4.1 renderer with orthographic 2D and perspective 3D cameras.
- Bounded Catch2 tests for numerical, topology, transfer, CLI, I/O, camera, and session invariants.

## Requirements

- A C++17 compiler
- CMake 3.16 or newer
- Ninja
- Git
- OpenGL development libraries for the optional workbench
- Just (optional)

CMake fetches pinned Eigen 3.4.0, Catch2 3.8.1, GLFW 3.4, and Dear ImGui 1.91.8 sources. The generated GLAD 2.0.8 OpenGL 4.1 loader is stored in `vendor/glad`.

## Build and Test

```bash
just build
just test
```

Or use CMake directly:

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build

cmake -S . -B build-dev -GNinja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DNUCLEAR_MPM_BUILD_TESTS=ON \
  -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
cmake --build build-dev --target nuclear_mpm_tests nuclear_mpm_solver
ctest --test-dir build-dev --output-on-failure
```

Routine validation is intentionally bounded; the test suite does not run stress or endurance simulations.

### Build options

| Option | Default | Purpose |
|---|---:|---|
| `NUCLEAR_MPM_BUILD_TESTS` | `ON` | Build Catch2 tests and CLI smoke tests |
| `NUCLEAR_MPM_BUILD_WORKBENCH` | `ON` | Build GLFW/OpenGL/Dear ImGui workbench |
| `NUCLEAR_MPM_ENABLE_OPENMP` | `OFF` | Enable optional partitioned P2G transfers when OpenMP is available |
| `NUCLEAR_MPM_ENABLE_SANITIZERS` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer on supported compilers |

AppleClang builds use UBSan alone because its arm64 macOS ASan runtime can hang during dynamic-shadow initialization; Clang and GCC platforms use combined ASan and UBSan.

Parallel execution remains opt-in and serial execution is the reference. Parallel scratch storage is capped at 64 MiB by default; worker count falls back before exceeding that budget.

## Interactive Workbench

```bash
just run
```

The workbench owns separate persistent 2D and 3D simulations. Switching dimensions does not discard either session's particles, elapsed time, pause state, or camera.

Controls include:

- Run/pause, single step, and simulation reset
- Live 2D/3D switching
- Solid, snow, and liquid scene presets
- Grid resolution, particle sampling, density, gravity, CFL, and timestep controls
- Per-face slip, sticky, and outflow boundaries
- Live mass, speed, deformation, substep, and outflow diagnostics

Camera controls:

- **2D:** middle-drag to pan, scroll to zoom
- **3D:** left-drag to orbit, middle-drag to pan, scroll to dolly

Camera input is disabled while Dear ImGui captures the mouse.

## Headless Solver

Show all arguments:

```bash
./build/nuclear_mpm_solver --help
```

Run one short 2D snow simulation and write a versioned snapshot:

```bash
./build/nuclear_mpm_solver \
  --dimension 2 \
  --material snow \
  --box-min 0.4,0.4 \
  --box-max 0.6,0.6 \
  --points-per-axis 16,16 \
  --gravity 0,-9.81 \
  --cells 64 \
  --duration 0.01 \
  --max-dt 0.0001 \
  --output tmp/snow-2d.nmpm
```

For 3D, provide three components for bounds, sampling counts, and gravity:

```bash
./build/nuclear_mpm_solver \
  --dimension 3 \
  --material solid \
  --box-min 0.4,0.4,0.4 \
  --box-max 0.6,0.6,0.6 \
  --points-per-axis 8,8,8 \
  --gravity 0,-9.81,0 \
  --cells 64 \
  --duration 0.001
```

## Embedding NuclearMPM

From the build tree, link `NuclearMPM::NuclearMPM`. The installed package has a public dependency on Eigen 3.4; install Eigen separately and ensure both package prefixes are discoverable through `CMAKE_PREFIX_PATH` (or `Eigen3_DIR`). Installed consumers can then use:

```cmake
find_package(NuclearMPM CONFIG REQUIRED)
target_link_libraries(my_simulation PRIVATE NuclearMPM::NuclearMPM)
```

Example:

```cpp
#include <nuclear_mpm/nuclear_mpm.hpp>

int main() {
    nuclear_mpm::SimulationConfig<2> config;
    config.cells_per_axis = 64;
    config.materials.add(nuclear_mpm::Snow<2>(400.0F, 0.2F));

    nuclear_mpm::RuntimeControls<2> controls;
    controls.gravity << 0.0F, -9.81F;

    auto particles = nuclear_mpm::sample_box_lattice<2>(
        nuclear_mpm::Index<2>{8, 8},
        nuclear_mpm::Vector<2>{0.4F, 0.4F},
        nuclear_mpm::Vector<2>{0.6F, 0.6F},
        1.0F,
        0,
        0xED553BU);

    nuclear_mpm::Simulation<2> simulation(config, controls, std::move(particles));
    const auto diagnostics = simulation.advance(1.0e-3F);
    return diagnostics.substep_count > 0 ? 0 : 1;
}
```

## Numerical Model

The core uses quadratic B-spline stencils and one dimension-generic `3^Dim` transfer path. Complete stencil support is validated before grid access. Fixed-corotated stress uses the Kirchhoff tensor

```text
τ = 2 μ (F - R) Fᵀ + λ (J - 1) J I
```

with an actual scaled identity for the volumetric term. Liquids use an isotropic bulk response with zero shear modulus. Particle velocity is never clipped to hide instability; `Simulation::advance` chooses substeps from transport speed, material wave speed, grid spacing, and the configured CFL factor.

## Installation

```bash
cmake -S . -B build-install -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNUCLEAR_MPM_BUILD_TESTS=OFF \
  -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
cmake --build build-install
cmake --install build-install --prefix /desired/prefix --component NuclearMPM
```

This installs only the core headers and CMake package metadata. Visualization dependencies remain outside the core package.
