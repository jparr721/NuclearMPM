# Production-Grade NuclearMPM Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current solver with a tested 2D/3D MLS-MPM library, bounded optional parallel execution, a corrected CLI, and a Dear ImGui workbench with persistent 2D/3D sessions and a controllable 3D camera.

**Architecture:** Build a deterministic serial numerical core first, with grid topology, stencil generation, constitutive models, boundaries, timestep policy, and diagnostics separated behind focused interfaces. Add bounded-memory OpenMP only after serial invariants pass. Keep GLFW, GLAD, OpenGL, and Dear ImGui in an optional workbench layer above the core library.

**Tech Stack:** C++17, CMake 3.16+, Ninja, Eigen 3.4.0, Catch2 3.8.1, optional OpenMP, GLFW 3.4, GLAD 2.0.8/OpenGL 4.1 Core, Dear ImGui 1.91.8.

## Global Constraints

- Work directly in the current `main` checkout; backward compatibility is not required.
- Use `PascalCase` for types, `snake_case` for functions/variables/fields, and trailing underscores for private members.
- The `NuclearMPM::NuclearMPM` core target must not depend on GLFW, GLAD, OpenGL, or Dear ImGui.
- Serial execution is the correctness reference and the default runtime mode.
- Parallel scratch memory is capped at 64 MiB by default and must fall back to fewer workers or serial execution before exceeding the cap.
- Routine tests use at most 64 particles and 16 simulation substeps; no stress or endurance tests are permitted.
- Debug tests retain Eigen assertions; Release builds must not be used to hide failed checks.
- Do not force `CMAKE_BUILD_TYPE`, `-Ofast`, or global compiler flags.
- Every numerical failure must include phase and particle/node context; do not silently clamp particle or grid velocity.
- Every Catch2 test case must carry the task tag used by its documented test command, such as `[math]` or `[camera]`.
- Commit each task only after its bounded test target passes.

---

## File Map

### Core library

- `include/nuclear_mpm/nuclear_mpm.hpp` — umbrella public include.
- `include/nuclear_mpm/types.hpp` — scalar, vector, matrix, index, and dimension aliases.
- `include/nuclear_mpm/math.hpp` — scaled identities, finite checks, signed SVD, and polar decomposition.
- `include/nuclear_mpm/particle.hpp` — particle state and invariant validation.
- `include/nuclear_mpm/grid.hpp` — grid nodes, topology, coordinate/index conversion, reset, and iteration.
- `include/nuclear_mpm/stencil.hpp` — quadratic B-spline support and generic `3^Dim` node iteration.
- `include/nuclear_mpm/material.hpp` — material parameter types, material table, stress, projection, and wave speed.
- `include/nuclear_mpm/boundary.hpp` — face policies and grid/particle boundary operations.
- `include/nuclear_mpm/config.hpp` — immutable topology/material configuration and mutable runtime controls.
- `include/nuclear_mpm/diagnostics.hpp` — bounded step metrics and numerical failure details.
- `include/nuclear_mpm/simulation.hpp` — serial MLS-MPM pipeline, adaptive substeps, and optional parallel dispatch.
- `include/nuclear_mpm/sampling.hpp` — cell-centered box lattice sampling with physical mass/volume.
- `include/nuclear_mpm/parallel.hpp` — worker/memory-budget selection and bounded OpenMP accumulators.

### CLI and I/O

- `src/solver.cpp` — validated 2D/3D command-line entry point.
- `src/snapshot_io.h` / `src/snapshot_io.cpp` — versioned metadata and particle/grid snapshot output.

### Workbench

- `src/camera.h` / `src/camera.cpp` — headless-testable orthographic and orbit-camera math.
- `src/visualizer.h` / `src/visualizer.cpp` — OpenGL window, 3D point rendering, domain rendering, and ImGui frame lifecycle.
- `src/simulation_session.h` / `src/simulation_session.cpp` — persistent 2D/3D session ownership and switching.
- `src/simulation_workbench.h` / `src/simulation_workbench.cpp` — Dear ImGui controls and simulation presentation.
- `src/example.cpp` — minimal workbench application entry point.

### Tests

- `tests/CMakeLists.txt`
- `tests/math_test.cpp`
- `tests/grid_test.cpp`
- `tests/stencil_test.cpp`
- `tests/particle_sampling_test.cpp`
- `tests/material_test.cpp`
- `tests/boundary_test.cpp`
- `tests/transfer_test.cpp`
- `tests/simulation_test.cpp`
- `tests/parallel_test.cpp`
- `tests/solver_cli_test.cpp`
- `tests/snapshot_io_test.cpp`
- `tests/camera_test.cpp`
- `tests/simulation_session_test.cpp`

---

### Task 1: Target-Scoped Build and Test Foundation

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `justfile`
- Create: `include/nuclear_mpm/nuclear_mpm.hpp`
- Create: `tests/CMakeLists.txt`
- Create: `tests/smoke_test.cpp`

**Interfaces:**
- Produces: `NuclearMPM::NuclearMPM` as an interface target requiring C++17 and Eigen.
- Produces: CMake options `NUCLEAR_MPM_BUILD_TESTS`, `NUCLEAR_MPM_BUILD_WORKBENCH`, `NUCLEAR_MPM_ENABLE_OPENMP`, and `NUCLEAR_MPM_ENABLE_SANITIZERS`.
- Produces: `nuclear_mpm_tests` Catch2 executable registered with CTest.

- [ ] **Step 1: Add a failing public-header smoke test**

```cpp
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/nuclear_mpm.hpp>

TEST_CASE("smoke: public header exposes the nuclear_mpm namespace", "[smoke]") {
    REQUIRE(nuclear_mpm::library_version_major == 1);
}
```

- [ ] **Step 2: Configure and verify the smoke test fails to compile**

Run:

```bash
cmake -S . -B build-dev -GNinja -DCMAKE_BUILD_TYPE=Debug \
  -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
cmake --build build-dev --target nuclear_mpm_tests
```

Expected: compilation fails because the new public API does not exist.

- [ ] **Step 3: Add target-scoped CMake and the minimal umbrella header**

```cpp
#pragma once

namespace nuclear_mpm {
inline constexpr int library_version_major = 1;
}
```

Pin Catch2 with `GIT_TAG v3.8.1`, preserve Eigen `3.4.0`, and remove forced build type/global optimization flags. Add `just test` using `ctest --test-dir build-dev --output-on-failure`.

- [ ] **Step 4: Run the bounded smoke test**

```bash
cmake --build build-dev --target nuclear_mpm_tests
ctest --test-dir build-dev --output-on-failure -R smoke
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt justfile include/nuclear_mpm tests
git commit -m "build: establish NuclearMPM library and tests"
```

---

### Task 2: Dimension-Safe Math Primitives

**Files:**
- Create: `include/nuclear_mpm/types.hpp`
- Create: `include/nuclear_mpm/math.hpp`
- Create: `tests/math_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `Scalar`, `Vector<Dim>`, `Matrix<Dim>`, `Index<Dim>`, and `RuntimeDimension { Two = 2, Three = 3 }`.
- Produces: `scaled_identity<Dim>(Scalar) -> Matrix<Dim>`.
- Produces: `filled_matrix<Dim>(Scalar) -> Matrix<Dim>`.
- Produces: `SignedSvd<Dim>` and `signed_svd<Dim>(const Matrix<Dim>&)`.
- Produces: `PolarDecomposition<Dim>` and `polar_decomposition<Dim>(const Matrix<Dim>&)`.

- [ ] **Step 1: Write failing 2D/3D identity and SVD tests**

```cpp
TEST_CASE("scaled identity is dimension complete") {
    REQUIRE(nuclear_mpm::scaled_identity<2>(2.0F).determinant() == Catch::Approx(4.0F));
    REQUIRE(nuclear_mpm::scaled_identity<3>(2.0F).determinant() == Catch::Approx(8.0F));
}

TEST_CASE("signed SVD reconstructs 2D and 3D matrices") {
    const auto result = nuclear_mpm::signed_svd<2>(test_matrix);
    REQUIRE((result.u * result.singular_values.asDiagonal() * result.v.transpose() - test_matrix).norm()
            < 1.0e-5F);
}
```

Include a reflected 2D matrix so the old hard-coded `col(2)` path would fail.

- [ ] **Step 2: Run only math tests and confirm failure**

```bash
cmake --build build-dev --target nuclear_mpm_tests
build-dev/tests/nuclear_mpm_tests "[math]"
```

Expected: compile failure for missing math interfaces.

- [ ] **Step 3: Implement dimension-safe helpers**

Use `Matrix<Dim>::Identity() * value`; never manually assign diagonal entries. In signed SVD, use `constexpr int last = Dim - 1` and return singular values as a vector rather than a partially initialized matrix. Reject non-finite inputs with `std::invalid_argument`.

- [ ] **Step 4: Run math tests**

```bash
build-dev/tests/nuclear_mpm_tests "[math]"
```

Expected: PASS with Eigen assertions active.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/types.hpp include/nuclear_mpm/math.hpp \
  include/nuclear_mpm/nuclear_mpm.hpp tests/math_test.cpp tests/CMakeLists.txt
git commit -m "feat: add dimension-safe MPM math primitives"
```

---

### Task 3: Grid Topology and Quadratic Stencils

**Files:**
- Create: `include/nuclear_mpm/grid.hpp`
- Create: `include/nuclear_mpm/stencil.hpp`
- Create: `tests/grid_test.cpp`
- Create: `tests/stencil_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`

**Interfaces:**
- Produces: `GridNode<Dim> { mass, momentum, velocity }`.
- Produces: `Grid<Dim>(int cells_per_axis)`, `nodes_per_axis()`, `node_count()`, `flatten(Index<Dim>)`, `coordinate(size_t)`, `node(Index<Dim>)`, and `reset()`.
- Produces: `QuadraticStencil<Dim>(position, inverse_cell_size, nodes_per_axis)`.
- Produces: `StencilNode<Dim> { index, offset, weight, grid_displacement, physical_displacement }`.
- Produces: `QuadraticStencil::for_each_node(callback)`.

- [ ] **Step 1: Write failing topology and kernel-invariant tests**

Test all coordinates of grids with 2 and 3 cells per axis; verify flatten/coordinate bijection and exact node counts. Test stencils at `0.5 * dx`, `0.5`, and `1 - 0.5 * dx`; verify all nodes are in `[0, nodes_per_axis)`, weights sum to one, and weighted grid displacement sums to zero.

- [ ] **Step 2: Run only grid/stencil tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[grid],[stencil]"
```

Expected: compile failure for missing grid/stencil types.

- [ ] **Step 3: Implement one dimension-generic topology/stencil path**

Compute the base as:

```cpp
const auto grid_position = position * inverse_cell_size;
base_ = (grid_position.array() - Scalar{0.5}).floor().template cast<int>();
fractional_ = grid_position - base_.template cast<Scalar>();
```

Validate `base[d] >= 0 && base[d] + 2 < nodes_per_axis`. Iterate offsets with a base-3 counter so 2D and 3D do not duplicate nested loops.

- [ ] **Step 4: Run bounded grid/stencil tests**

```bash
build-dev/tests/nuclear_mpm_tests "[grid],[stencil]"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/grid.hpp include/nuclear_mpm/stencil.hpp \
  include/nuclear_mpm/nuclear_mpm.hpp tests/grid_test.cpp tests/stencil_test.cpp
git commit -m "feat: centralize grid topology and quadratic stencils"
```

---

### Task 4: Particle State and Physical Box Sampling

**Files:**
- Create: `include/nuclear_mpm/particle.hpp`
- Create: `include/nuclear_mpm/sampling.hpp`
- Create: `tests/particle_sampling_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`

**Interfaces:**
- Produces: `Particle<Dim>` with `position`, `velocity`, `elastic_deformation`, `affine_velocity`, `plastic_volume_ratio`, `mass`, `reference_volume`, `material_id`, and `user_tag`.
- Produces: `validate_particle(const Particle<Dim>&, size_t particle_index)`.
- Produces: `sample_box_lattice(Index<Dim> points_per_axis, Vector<Dim> min_corner, Vector<Dim> max_corner, Scalar density, size_t material_id, uint32_t user_tag) -> std::vector<Particle<Dim>>`.

- [ ] **Step 1: Write failing sampling tests**

Verify a `2 x 3` 2D lattice returns six cell-centered particles, all strictly inside the requested box, with total reference volume equal to box area and total mass equal to `density * area`. Repeat with `2 x 2 x 2` in 3D. Verify zero density, reversed bounds, or non-positive counts throw `std::invalid_argument`.

- [ ] **Step 2: Run sampling tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[sampling]"
```

Expected: compile failure for missing particle/sampling interfaces.

- [ ] **Step 3: Implement validated cell-centered sampling**

Use per-axis fractions `(index + 0.5) / points_per_axis`, calculate `reference_volume = box_measure / particle_count`, and `mass = density * reference_volume`. Initialize deformation with `scaled_identity<Dim>(1)`.

- [ ] **Step 4: Run sampling tests**

```bash
build-dev/tests/nuclear_mpm_tests "[sampling]"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/particle.hpp include/nuclear_mpm/sampling.hpp \
  include/nuclear_mpm/nuclear_mpm.hpp tests/particle_sampling_test.cpp
git commit -m "feat: add physical particle state and box sampling"
```

---

### Task 5: Constitutive Material Models

**Files:**
- Create: `include/nuclear_mpm/material.hpp`
- Create: `tests/material_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`

**Interfaces:**
- Produces: `FixedCorotatedSolid<Dim>`, `Snow<Dim>`, and `WeaklyCompressibleLiquid<Dim>` parameter types.
- Produces: `Material<Dim>` variant and `MaterialTable<Dim>`.
- Produces: `kirchhoff_stress(material, particle) -> Matrix<Dim>`.
- Produces: `project_deformation(const Material<Dim>&, const Matrix<Dim>& trial_deformation, Scalar& plastic_volume_ratio) -> Matrix<Dim>`.
- Produces: `characteristic_wave_speed(material, density) -> Scalar`.

- [ ] **Step 1: Write failing constitutive tests**

For `F = 1.1 I`, require solid stress off-diagonal entries to be zero. For pure shear with `det(F) = 1`, require liquid stress to have zero shear entries. Verify snow singular values are projected to `[1 - 2.5e-2, 1 + 4.5e-3]` and plastic volume remains `[0.6, 20]`. Verify invalid Young's modulus, Poisson ratio, bulk modulus, and density throw.

- [ ] **Step 2: Run material tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[material]"
```

Expected: compile failure for missing material interfaces.

- [ ] **Step 3: Implement model-specific stress and projection**

Implement solid/snow Kirchhoff stress exactly as:

```cpp
return Scalar{2} * mu * (f - rotation) * f.transpose()
     + lambda * (j - Scalar{1}) * j * scaled_identity<Dim>(Scalar{1});
```

Implement liquid stress as isotropic bulk response with zero shear, and project its deformation to `pow(J, 1 / Dim) * I`. Use `std::visit` for closed variant dispatch.

- [ ] **Step 4: Run material and math tests**

```bash
build-dev/tests/nuclear_mpm_tests "[material],[math]"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/material.hpp include/nuclear_mpm/nuclear_mpm.hpp tests/material_test.cpp
git commit -m "feat: implement solid snow and liquid materials"
```

---

### Task 6: Configuration, Errors, and Boundary Policies

**Files:**
- Create: `include/nuclear_mpm/config.hpp`
- Create: `include/nuclear_mpm/boundary.hpp`
- Create: `include/nuclear_mpm/diagnostics.hpp`
- Create: `tests/boundary_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`

**Interfaces:**
- Produces: `BoundaryMode { Slip, Sticky, Outflow }` and `DomainBoundaries<Dim>` with lower/upper face arrays plus `boundary_layers` and `outflow_deletion_layers`.
- Produces: `SimulationConfig<Dim>` with `cells_per_axis`, `max_time_step`, `cfl_factor`, materials, boundaries, and `parallel_config`.
- Produces: `RuntimeControls<Dim>` with gravity and playback-independent live values plus `validate_runtime_controls(...)`.
- Produces: `SimulationError` carrying phase, entity index, position/index, and explanatory text.
- Produces: `StepDiagnostics<Dim>` and `apply_grid_boundaries(...)`.

- [ ] **Step 1: Write failing boundary/configuration tests**

Verify slip removes only outward normal velocity, sticky removes all velocity, neither changes node mass, and outflow does not alter grid velocity. Verify invalid resolution, timestep, CFL factor, gravity, and boundary width throw.

- [ ] **Step 2: Run boundary tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[boundary],[config]"
```

Expected: compile failure for missing configuration/boundary types.

- [ ] **Step 3: Implement validated configuration and per-face policies**

Use `std::array<BoundaryMode, Dim>` for lower and upper faces. Apply only the component normal to a slip face; apply all components for sticky. Keep mass and momentum accounting intact.

- [ ] **Step 4: Run boundary/configuration tests**

```bash
build-dev/tests/nuclear_mpm_tests "[boundary],[config]"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/config.hpp include/nuclear_mpm/boundary.hpp \
  include/nuclear_mpm/diagnostics.hpp include/nuclear_mpm/nuclear_mpm.hpp tests/boundary_test.cpp
git commit -m "feat: add validated simulation and boundary policies"
```

---

### Task 7: Deterministic Serial MLS-MPM Pipeline

**Files:**
- Create: `include/nuclear_mpm/simulation.hpp`
- Create: `tests/transfer_test.cpp`
- Create: `tests/simulation_test.cpp`
- Modify: `include/nuclear_mpm/nuclear_mpm.hpp`

**Interfaces:**
- Produces: `Simulation<Dim>(SimulationConfig<Dim>, RuntimeControls<Dim>, std::vector<Particle<Dim>>)`. 
- Produces: `substep(Scalar dt) -> StepDiagnostics<Dim>`.
- Produces: immutable `particles()`, `grid()`, `config()`, and `runtime_controls()` plus `set_runtime_controls(RuntimeControls<Dim>)` with validation before assignment.
- Consumes: grid, stencil, particle, material, and boundary contracts from Tasks 2–6.

- [ ] **Step 1: Write failing transfer invariants**

Use at most eight particles. Verify P2G grid mass equals particle mass, force-free transfer preserves linear momentum, a resting particle remains at rest, uniform velocity is reproduced, and a 2D affine velocity field reconstructs within tolerance. Add one non-empty 3D one-substep test.

- [ ] **Step 2: Run transfer tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[transfer],[simulation]"
```

Expected: compile failure for missing `Simulation`.

- [ ] **Step 3: Implement explicit serial phases**

Implement reset, P2G accumulation, momentum normalization, body force, boundaries, G2P, advection, and material deformation projection as separate private functions. Build the P2G affine matrix as:

```cpp
const Matrix<Dim> stress_impulse =
    -dt * particle.reference_volume * Scalar{4} * inverse_cell_size_ * inverse_cell_size_
    * kirchhoff_stress(material, particle);
const Matrix<Dim> affine_momentum = stress_impulse + particle.mass * particle.affine_velocity;
```

Every P2G/G2P access must come from `QuadraticStencil`; no manual flatten formulas are allowed.

- [ ] **Step 4: Run bounded transfer tests**

```bash
build-dev/tests/nuclear_mpm_tests "[transfer],[simulation]"
```

Expected: PASS; no test exceeds one substep in this task.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/simulation.hpp include/nuclear_mpm/nuclear_mpm.hpp \
  tests/transfer_test.cpp tests/simulation_test.cpp
git commit -m "feat: implement deterministic serial MLS-MPM"
```

---

### Task 8: CFL Substepping, Outflow, and Diagnostics

**Files:**
- Modify: `include/nuclear_mpm/simulation.hpp`
- Modify: `include/nuclear_mpm/diagnostics.hpp`
- Modify: `tests/simulation_test.cpp`

**Interfaces:**
- Produces: `advance(Scalar duration) -> StepDiagnostics<Dim>`.
- Produces: `stable_time_step() -> Scalar` using transport speed and material wave speed.
- Produces: explicit outflow removal counts and mass in diagnostics.

- [ ] **Step 1: Write failing timestep/diagnostic tests**

Verify the selected timestep is `<= max_time_step` and satisfies `dt <= cfl * dx / (max_transport_speed + max_wave_speed)`. Verify a bounded eight-substep free-fall result, finite deformation determinants, contextual rejection of non-finite state, and explicit outflow removal before stencil support becomes invalid.

- [ ] **Step 2: Run simulation tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[simulation]"
```

Expected: failures for missing adaptive advance and diagnostics.

- [ ] **Step 3: Implement bounded adaptive advance**

Recompute the stability estimate before each substep, cap the final substep to remaining duration, reject non-positive/non-finite estimates, remove outflow particles at the configured deletion plane, and aggregate diagnostics without changing physical state.

- [ ] **Step 4: Run bounded simulation tests**

```bash
build-dev/tests/nuclear_mpm_tests "[simulation]"
```

Expected: PASS; each test uses at most 16 particles and eight substeps.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/simulation.hpp include/nuclear_mpm/diagnostics.hpp tests/simulation_test.cpp
git commit -m "feat: add CFL stepping and numerical diagnostics"
```

---

### Task 9: Bounded-Memory Optional OpenMP P2G

**Files:**
- Create: `include/nuclear_mpm/parallel.hpp`
- Create: `tests/parallel_test.cpp`
- Modify: `include/nuclear_mpm/config.hpp`
- Modify: `include/nuclear_mpm/simulation.hpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `ParallelConfig { enabled = false, max_workers = 4, scratch_memory_budget_bytes = 64 * 1024 * 1024 }`.
- Produces: `select_parallel_worker_count(grid_bytes, requested_workers, budget_bytes) -> size_t`.
- Produces: partition-local mass/momentum accumulation with fixed-order reduction.

- [ ] **Step 1: Write failing memory-budget and parity tests**

Verify worker selection never returns a count whose scratch grids exceed the budget and returns one for a grid too large for two copies. With OpenMP available, compare serial and parallel P2G mass/momentum for at most 32 particles within `1e-5`; otherwise mark only the parity case skipped while still testing budget selection.

- [ ] **Step 2: Run parallel tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[parallel]"
```

Expected: compile failure for missing parallel interfaces.

- [ ] **Step 3: Implement budgeted partition accumulation**

Use no more than `min(max_workers, floor(budget / grid_bytes))` partitions, with one meaning serial execution. Reduce partitions into the primary grid in ascending partition then ascending node order. Only define `NUCLEAR_MPM_HAS_OPENMP` when `find_package(OpenMP)` succeeds and the option is enabled.

- [ ] **Step 4: Run parallel and transfer tests**

```bash
build-dev/tests/nuclear_mpm_tests "[parallel],[transfer]"
```

Expected: PASS without allocating more than the configured test budget.

- [ ] **Step 5: Commit**

```bash
git add include/nuclear_mpm/parallel.hpp include/nuclear_mpm/config.hpp \
  include/nuclear_mpm/simulation.hpp tests/parallel_test.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: add bounded-memory OpenMP transfer"
```

---

### Task 10: Validated 2D/3D CLI and Snapshot I/O

**Files:**
- Create: `src/solver_cli.h`
- Create: `src/solver_cli.cpp`
- Replace: `src/solver.cpp`
- Create: `src/snapshot_io.h`
- Create: `src/snapshot_io.cpp`
- Create: `tests/solver_cli_test.cpp`
- Create: `tests/snapshot_io_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `SolverOptions parse_solver_options(int argc, char** argv)` using `RuntimeDimension` with `--dimension`, `--material`, `--box-min`, `--box-max`, `--points-per-axis`, `--cells`, `--duration`, `--max-dt`, `--gravity`, and `--output`.
- Produces: `int run_solver(const SolverOptions&)` with explicit 2D/3D dispatch.
- Produces: `write_snapshot(path, simulation_time, config, particles, diagnostics)` for both dimensions.
- Produces a versioned text header followed by dimension-aware particle records; grid indexing remains inside `Grid`.

- [ ] **Step 1: Write failing CLI/parser and snapshot round-trip tests**

Verify unsupported dimensions/materials fail, vector argument lengths match dimension, a non-empty 3D configuration is produced, and a four-particle snapshot round-trips metadata and fields. Use temporary files only.

- [ ] **Step 2: Run I/O tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[io],[cli]"
```

Expected: compile failure for missing snapshot interfaces.

- [ ] **Step 3: Implement explicit geometry arguments and versioned output**

Keep CLI parsing separate from simulation construction. Do not call X/Y arguments cube positions when they are bounds. Reject extra or missing coordinate components before constructing particles.

- [ ] **Step 4: Run I/O tests and one-step CLI smoke tests**

```bash
build-dev/tests/nuclear_mpm_tests "[io],[cli]"
ctest --test-dir build-dev --output-on-failure -R "solver_(2d|3d)_one_step"
```

Expected: PASS; smoke tests execute exactly one bounded step.

- [ ] **Step 5: Commit**

```bash
git add src/solver_cli.h src/solver_cli.cpp src/solver.cpp src/snapshot_io.h src/snapshot_io.cpp \
  tests/solver_cli_test.cpp tests/snapshot_io_test.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: rebuild solver CLI and snapshot output"
```

---

### Task 11: Native 3D Renderer and Camera

**Files:**
- Create: `src/camera.h`
- Create: `src/camera.cpp`
- Replace: `src/visualizer.h`
- Replace: `src/visualizer.cpp`
- Create: `tests/camera_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `OrthographicCamera` with `pan(delta)`, `zoom(delta)`, `reset(domain_bounds)`, `view_matrix()`, and `projection_matrix(aspect_ratio)`.
- Produces: `OrbitCamera` with `orbit(delta)`, `pan(delta)`, `dolly(delta)`, `reset(domain_bounds)`, `view_matrix()`, and `projection_matrix(aspect_ratio)`.
- Produces: `CameraInput filter_camera_input(const CameraInput&, bool ui_wants_mouse)`.
- Produces: `RenderPoint { float x, y, z; uint32_t color; }` and `RenderScene { points, RuntimeDimension dimension, domain_min, domain_max }`.
- Produces: `Visualizer::begin_frame()`, `draw_scene(scene, view_projection)`, `end_frame()`, `should_close()`, and normalized pointer/scroll input.

- [ ] **Step 1: Write failing headless camera tests**

Verify orthographic and perspective resets frame the unit box, orbit preserves camera-target distance, pan translates eye and target equally, dolly clamps positive distance, projection changes with aspect ratio, all matrices remain finite, and `filter_camera_input` returns neutral input when UI capture is true.

- [ ] **Step 2: Run camera tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[camera]"
```

Expected: compile failure for missing camera types.

- [ ] **Step 3: Implement camera math and 3D OpenGL rendering**

Change the vertex shader input to `vec3`, add a `mat4 view_projection` uniform, render the 2D domain as a rectangle and the 3D domain as twelve box edges, and enable depth testing only for 3D. Keep OpenGL objects RAII-owned.

- [ ] **Step 4: Run camera tests and compile the optional visualizer**

```bash
build-dev/tests/nuclear_mpm_tests "[camera]"
cmake --build build-dev --target nuclear_mpm_visualizer
```

Expected: PASS and compile success; do not launch a window in automated validation.

- [ ] **Step 5: Commit**

```bash
git add src/camera.h src/camera.cpp src/visualizer.h src/visualizer.cpp \
  tests/camera_test.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: add native 3D rendering and orbit camera"
```

---

### Task 12: Persistent Sessions and Dear ImGui Workbench

**Files:**
- Create: `src/simulation_session.h`
- Create: `src/simulation_session.cpp`
- Create: `src/simulation_workbench.h`
- Create: `src/simulation_workbench.cpp`
- Replace: `src/example.cpp`
- Create: `tests/simulation_session_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `RuntimeDimension` from the core type layer.
- Produces: `SimulationSessionController` owning persistent `Simulation<2>` and `Simulation<3>`, separate cameras, elapsed times, and pause states.
- Produces: `switch_dimension`, `advance_active`, `single_step_active`, `reset_active`, and `active_render_scene`.
- Produces: `SimulationWorkbench::run()` and ImGui panels for simulation, scene, boundaries, camera, and diagnostics.

- [ ] **Step 1: Write failing session-preservation tests**

Create tiny sessions with at most eight particles. Advance 2D once, switch to 3D and advance once, switch back, and require each session's elapsed time and particle positions to retain its own state. Verify staged topology changes do not apply until `reset_active`.

- [ ] **Step 2: Run session tests and confirm failure**

```bash
build-dev/tests/nuclear_mpm_tests "[session]"
```

Expected: compile failure for missing controller.

- [ ] **Step 3: Implement controller and ImGui workbench**

Pin Dear ImGui `v1.91.8` and compile `imgui.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp`, `imgui_impl_glfw.cpp`, and `imgui_impl_opengl3.cpp` into a dedicated target. Route orbit/pan/dolly input only when `ImGuiIO::WantCaptureMouse` is false. Expose run/pause, single step/frame, reset, material/scene preset, runtime values, staged topology, per-face boundaries, diagnostics, and 2D/3D switching.

- [ ] **Step 4: Run session tests and compile the workbench**

```bash
build-dev/tests/nuclear_mpm_tests "[session]"
cmake --build build-dev --target nuclear_mpm_example
```

Expected: PASS and compile success; no unattended GUI launch.

- [ ] **Step 5: Commit**

```bash
git add src/simulation_session.h src/simulation_session.cpp \
  src/simulation_workbench.h src/simulation_workbench.cpp src/example.cpp \
  tests/simulation_session_test.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat: add ImGui 2D and 3D simulation workbench"
```

---

### Task 13: Packaging, Sanitizers, Documentation, and Legacy Removal

**Files:**
- Delete: `src/nclr.h`
- Delete: `src/nclr_math.h`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `justfile`
- Create: `cmake/NuclearMPMConfig.cmake.in`
- Create: `cmake/NuclearMPMWarnings.cmake`
- Create: `cmake/NuclearMPMSanitizers.cmake`
- Create: `tests/package_consumer/CMakeLists.txt`
- Create: `tests/package_consumer/main.cpp`
- Create: `cmake/RunPackageConsumerTest.cmake`

**Interfaces:**
- Produces install/export support for `find_package(NuclearMPM CONFIG REQUIRED)` and `NuclearMPM::NuclearMPM`.
- Produces `just test`, `just test-sanitize`, `just build-workbench`, and `just run`.
- Removes all obsolete names and duplicate solver paths.

- [ ] **Step 1: Add failing package-consumer and legacy-absence checks**

Add a tiny CMake consumer fixture that installs NuclearMPM to a temporary prefix, finds the package, includes `<nuclear_mpm/nuclear_mpm.hpp>`, and instantiates 2D and 3D public types. Add a CMake check that no production source includes `nclr.h` or `nclr_math.h`.

- [ ] **Step 2: Run package checks and confirm failure**

```bash
ctest --test-dir build-dev --output-on-failure -R "package_consumer|legacy_headers"
```

Expected: FAIL before install/export and cleanup are implemented.

- [ ] **Step 3: Complete packaging and remove legacy implementation**

Install `include/nuclear_mpm`, export the namespaced target, add compiler-specific warning flags, add opt-in ASan/UBSan flags, document the corrected API/CLI/workbench/camera controls, and delete old headers only after every source has migrated.

- [ ] **Step 4: Run final bounded verification**

```bash
cmake -S . -B build-dev -GNinja -DCMAKE_BUILD_TYPE=Debug \
  -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=ON
cmake --build build-dev
ctest --test-dir build-dev --output-on-failure

cmake -S . -B build-sanitize -GNinja -DCMAKE_BUILD_TYPE=Debug \
  -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF \
  -DNUCLEAR_MPM_ENABLE_SANITIZERS=ON
cmake --build build-sanitize --target nuclear_mpm_tests
ctest --test-dir build-sanitize --output-on-failure
```

Expected: all bounded tests pass; no sanitizer findings. Do not run the interactive workbench or any long simulation automatically.

- [ ] **Step 5: Commit**

```bash
git add -A CMakeLists.txt README.md justfile cmake include src tests
git commit -m "release: complete production NuclearMPM rewrite"
```

---

## Delivery Tracks

1. **Numerical core:** Tasks 1–8. This is the first release gate and must be correct before any optimization or UI work.
2. **Parallelism and tooling:** Tasks 9–10. These consume only public core interfaces.
3. **Visualization workbench:** Tasks 11–12. Camera math and session state remain independently testable without OpenGL windows.
4. **Release cleanup:** Task 13. Packaging and deletion happen only after all callers migrate.

## Stop Conditions

Pause implementation and report rather than guessing if:

- a primary reference contradicts the specified MLS-MPM equations;
- a serial conservation invariant cannot be met within a justified floating-point tolerance;
- 3D requires a dimension-specific duplicate transfer path;
- optional parallel execution cannot stay under its configured memory budget;
- a requested topology/material change cannot preserve an existing session without reset;
- sanitizer output identifies a defect outside the currently executing task.
