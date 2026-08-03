# Production-Grade NuclearMPM Redesign

## Status

Approved direction: replace the current compact solver with a clean, correctness-first MLS-MPM library. Backward compatibility is not required. The redesign must support real 2D and 3D simulations, explicit invariants, bounded deterministic tests, and optional safe parallel execution.

## Goals

- Implement a mathematically correct MLS-MPM/APIC transfer pipeline.
- Support 2D and 3D through one dimension-generic implementation.
- Provide solid, snow, and liquid constitutive models with model-appropriate behavior.
- Make invalid grid stencils unrepresentable during normal simulation operation.
- Replace velocity clipping with explicit timestep/CFL control.
- Separate numerical state, topology, materials, boundaries, diagnostics, examples, and I/O.
- Provide deterministic unit and short integration tests that catch the existing failures.
- Offer an installable CMake target with warnings, sanitizers, and optional OpenMP configured correctly.
- Provide a Dear ImGui simulation workbench that can switch live between persistent 2D and 3D sessions.
- Render native 3D particle positions with an interactive orbit/pan/dolly camera.

## Non-Goals

- Preserve source or binary compatibility with the existing API.
- Add GPU execution, sparse grids, CPIC, rigid-body coupling, fracture, or distributed execution.
- Optimize before the serial reference implementation is verified.
- Use long-running simulations as routine validation.

## Architecture

### Public layout

The library will expose an umbrella header and focused components:

```text
include/nuclear_mpm/
  nuclear_mpm.hpp
  math.hpp
  particle.hpp
  grid.hpp
  stencil.hpp
  material.hpp
  boundary.hpp
  simulation.hpp
  diagnostics.hpp
  sampling.hpp
src/
  solver.cpp
  example.cpp
  visualizer.*
  camera.*
  simulation_workbench.*
tests/
  math_test.cpp
  stencil_test.cpp
  grid_test.cpp
  material_test.cpp
  transfer_test.cpp
  boundary_test.cpp
  simulation_test.cpp
```

Dimension-dependent components will remain templates in the `nuclear_mpm` namespace. The initial scalar type will remain `float`, with aliases localized so a future scalar template does not require another structural rewrite. Types will use `PascalCase`; functions, variables, and fields will use `snake_case`; private data members will use a trailing underscore. Physical quantities will include units or coordinate-space meaning in their names whenever ambiguity is possible.

### Core state

`Particle<Dim>` will contain only simulation state:

- position
- velocity
- elastic deformation gradient
- APIC affine velocity field
- plastic volume ratio
- mass
- reference volume
- material identifier
- optional user tag

Rendering color will not be part of the physics model.

`GridNode<Dim>` will hold mass, momentum, and velocity as distinct fields. Momentum and velocity will never share the same variable at different phases.

`Grid<Dim>` will own topology and storage. It will define `cellsPerAxis`, `nodesPerAxis = cellsPerAxis + 1`, checked coordinate flattening, coordinate iteration, and reset behavior. No caller will manually duplicate flattening formulas.

### Quadratic B-spline stencil

`QuadraticStencil<Dim>` will be the only implementation of particle-grid neighborhood selection. It will:

1. compute the base coordinate with element-wise `floor`, not integer truncation;
2. compute fractional grid coordinates and one-dimensional kernel weights;
3. iterate all `3^Dim` node offsets through one dimension-generic iterator;
4. validate the complete support invariant:
   `0 <= base[d] && base[d] + 2 < nodesPerAxis`;
5. provide physical and grid-space node displacements with explicit names.

The simulation will validate initial particles before stepping. A stencil violation during stepping will produce a contextual simulation error containing particle index, position, base coordinate, and domain limits. Assertions may supplement this but will not be the only protection.

### Materials

Material behavior will be isolated behind a small constitutive interface with:

- Lamé/material parameter validation;
- Kirchhoff stress evaluation;
- deformation projection/update after G2P;
- characteristic wave-speed estimation for timestep control.

Initial models:

- `FixedCorotatedSolid`: elastic fixed-corotated stress.
- `Snow`: fixed-corotated elasticity, plastic singular-value projection, and plastic hardening.
- `WeaklyCompressibleLiquid`: zero shear modulus and an isotropic equation-of-state/volumetric response; it must not reuse solid shear behavior.

The fixed-corotated Kirchhoff stress will be expressed directly as:

```text
τ = 2 μ (F - R) Fᵀ + λ (J - 1) J I
```

The identity term must be a scaled identity, never an all-elements constant matrix. Helpers will use names such as `scaled_identity`, `filled_matrix`, `polar_decomposition`, and `signed_svd` to prevent semantic confusion.

A `MaterialTable<Dim>` will own validated material variants, and particles will select one by material identifier. Dispatch will use a closed `std::variant` of supported constitutive models rather than virtual calls inside particle loops. New material implementations must satisfy the same stress, projection, and wave-speed contract.

### Transfer pipeline

`Simulation::substep(dt)` will have explicit phases:

1. reset grid;
2. particle-to-grid mass and momentum transfer;
3. convert grid momentum to velocity;
4. apply body forces;
5. apply boundary conditions;
6. grid-to-particle velocity and affine-field transfer;
7. advect particles and update/project deformation;
8. validate finite state and collect diagnostics.

The fused P2G affine matrix will have a name reflecting what it is; it will not be called first Piola stress. P2G and G2P will share the same stencil implementation.

The serial transfer path will be the reference implementation. Optional OpenMP P2G will use thread-local grids followed by deterministic reduction rather than unsynchronized shared writes. OpenMP pragmas will only be compiled when CMake finds and enables OpenMP.

### Timestep policy

Arbitrary component-wise velocity clipping will be removed. `Simulation::advance(duration)` will choose bounded substeps from:

- requested maximum timestep;
- grid spacing;
- maximum material wave speed;
- maximum particle/grid transport velocity;
- a configurable CFL safety factor.

If a valid timestep cannot be produced because state is non-finite or parameters are invalid, the simulation will fail with diagnostic context rather than altering physics silently.

### Boundaries

Boundary behavior will be configured per domain face:

- `Slip`: remove only outward normal velocity;
- `Sticky`: zero the full grid velocity;
- `Outflow`: permit outward velocity and remove escaped particles at a deletion plane that is still inside valid quadratic-stencil support.

Boundary operations will not erase grid mass. Default closed-box boundaries will occupy enough grid layers to keep the quadratic stencil valid. Outflow particle removal will be explicit in diagnostics and mass accounting. The lower, upper, and tangential behavior will be independently testable.

### Configuration and errors

`SimulationConfig` will validate:

- 2D or 3D instantiation;
- positive grid resolution and timestep limits;
- CFL factor in a safe range;
- finite gravity;
- valid material parameters, including Poisson-ratio restrictions;
- positive particle mass/reference volume;
- finite initial particle state;
- complete initial stencil support.

Invalid configuration will throw `std::invalid_argument`. Runtime numerical failures will throw a dedicated `SimulationError` with phase and particle/node context.

### Diagnostics

Each advance will return or expose `StepDiagnostics` containing:

- substep count and actual timestep range;
- total particle/grid mass;
- linear momentum;
- maximum particle and grid speed;
- minimum/maximum deformation determinant;
- particle bounds;
- non-finite and stencil-violation counters.

Diagnostics are observational and must not mutate the simulation.

### Sampling, CLI, and I/O

Sampling helpers will use explicit geometry names and vector bounds, for example `sample_box_lattice(points_per_axis, min_corner, max_corner, density, material_id)`. They will derive per-particle reference volume and mass from the sampled region and density.

The CLI will replace ambiguous `cubeN-x/y` options with explicit box bounds and support a real 3D path. Unsupported dimensions and material names will fail immediately. Grid serialization will use the grid topology API instead of reimplementing flattening. Output files will carry enough metadata to interpret dimensions, resolution, timestep, and field layout.

### Interactive workbench and visualizer

The example application will become a Dear ImGui workbench rather than a render-only loop. Dear ImGui will be fetched at a pinned revision and compiled into the visualizer application target; it will not become a dependency of the core MPM library.

The workbench will own one persistent 2D session and one persistent 3D session through an explicit runtime controller. Switching dimensions will change the active session without discarding the inactive session's particles, elapsed time, camera, or pause state. Reset and recreation will be explicit actions because resolution and material-layout changes cannot safely mutate an existing simulation in place.

The control surface will include:

- 2D/3D session selector;
- run, pause, single-substep, single-frame, and reset controls;
- material and scene presets;
- gravity, CFL factor, maximum timestep, and playback-speed controls;
- box-sampling controls with explicit minimum/maximum corners and density;
- boundary-mode controls per domain face;
- live step diagnostics, particle count, timing, and numerical warnings.

Changes that can safely apply live, such as gravity or pause state, will do so through a separate runtime-controls object. Topology, dimension, and material-layout changes will be staged and applied only through an explicit reset/rebuild action. The UI will label that distinction instead of silently resetting the simulation.

The renderer will accept renderer-neutral 3D positions and colors. The 2D mode will use an orthographic camera and render particles in the XY plane. The 3D mode will use a perspective camera with:

- left-drag orbit around a target;
- middle-drag pan in the camera plane;
- scroll dolly/zoom;
- frame-selection and reset-camera actions;
- viewport resize handling and stable near/far planes derived from the simulation domain.

Camera input will be ignored whenever Dear ImGui reports mouse or keyboard capture. Camera and projection math will be independent of OpenGL so it can be unit tested without opening a window. Rendering resources will remain RAII-owned and visualization will remain optional at build time.

The particle shader will consume `vec3` positions and a view-projection matrix. The domain outline will render as a rectangle in 2D and a wireframe box in 3D. Depth testing will be enabled only for the 3D scene; point size and color behavior will remain configurable by the visualizer.

### Build and packaging

CMake will:

- stop forcing Release mode and global `-Ofast` flags;
- expose a namespaced library target, `NuclearMPM::NuclearMPM`;
- use target-scoped warning and optimization flags;
- provide options for examples, tests, sanitizers, and OpenMP;
- install headers and export package configuration;
- compile both 2D and 3D test targets;
- keep GLFW, GLAD, OpenGL, and Dear ImGui dependencies outside the core target;
- pin Dear ImGui and compile its GLFW/OpenGL backend sources in a dedicated UI target;
- use a pinned lightweight unit-test dependency integrated through CTest.

## Testing Strategy

All routine tests must be bounded and small. No stress tests or long simulations are part of this plan.

### Unit invariants

- 2D and 3D scaled identities have the correct diagonal and determinant.
- Signed SVD reconstructs its input without dimension-specific invalid indexing.
- Polar decomposition produces a proper rotation where defined.
- Quadratic weights satisfy partition of unity and first-moment reproduction.
- Lower and upper valid stencils select only valid nodes.
- Grid flatten/unflatten is bijective over every node.
- Uniform dilation produces diagonal isotropic stress with no artificial shear.
- Liquid stress has no shear response.
- Snow projection keeps singular values and plastic volume in configured bounds.

### Transfer invariants

- P2G conserves particle mass.
- Force-free P2G/grid/G2P conserves linear momentum within tolerance.
- A particle at rest remains at rest without forces.
- Uniform velocity is reproduced by G2P.
- An affine velocity field is reproduced within tolerance.

### Short integration tests

- Gravity produces expected short-time free fall away from boundaries.
- Slip boundaries remove only outward normal motion.
- Sticky boundaries remove all contact velocity.
- A small 2D and 3D block remains finite and stencil-valid for a small fixed number of substeps.
- Serial and OpenMP paths agree within documented floating-point tolerance.
- Switching between 2D and 3D preserves both sessions' simulation state.
- Camera orbit, pan, dolly, reset, and projection math pass headless deterministic tests.
- ImGui input capture prevents camera movement while interacting with controls.

### Tooling checks

- Debug tests with Eigen assertions enabled.
- AddressSanitizer and UndefinedBehaviorSanitizer targets.
- ThreadSanitizer coverage for the optional parallel transfer where supported.
- Warning-clean builds on Clang and GCC.

## Delivery Sequence

1. Establish the new library target and bounded test harness.
2. Implement dimension-safe math, grid topology, and stencil primitives.
3. Implement and verify constitutive models.
4. Implement the serial transfer pipeline and conservation tests.
5. Add boundary policies, configuration validation, CFL substepping, and diagnostics.
6. Add a race-free optional OpenMP backend and parity tests.
7. Migrate sampling, solver CLI, and output code.
8. Upgrade the renderer to native 2D/3D rendering and add tested camera math.
9. Add the Dear ImGui workbench, persistent dimension switching, controls, and diagnostics.
10. Add installation/export support, sanitizers, documentation, and CI-ready presets.
11. Delete the obsolete implementation and compatibility names.

## Acceptance Criteria

- All unit and bounded integration tests pass in Debug and Release.
- ASan/UBSan runs report no issues.
- Both 2D and 3D simulations execute non-empty bounded examples.
- No manual grid-index formulas exist outside `Grid`/`QuadraticStencil`.
- No velocity clipping is used to maintain stencil validity.
- No unsynchronized shared P2G writes exist.
- Isotropic deformation creates no artificial shear.
- Liquid has no elastic shear response.
- Public names describe their physical or algorithmic meaning accurately.
- Existing OOB, SVD, 3D identity, grid-dump stride, CLI-dimension, and debug-build defects have regression coverage.
- The workbench switches between non-empty 2D and 3D sessions without resetting either one.
- The 3D renderer uses actual 3D positions, depth testing, a perspective projection, and an interactive orbit/pan/dolly camera.
- The core library builds and tests without any visualization or Dear ImGui dependency.
