# Workbench Grid and Boundary Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the workbench's 64-cell default free fall and render testable wireframe grid and boundary guides for the active simulation.

**Architecture:** Keep simulation physics unchanged. Extend `RenderScene` with active-grid metadata, generate overlay lines in a new OpenGL-independent module, and let the existing visualizer upload those lines before particles. Workbench checkboxes control visibility without rebuilding the simulation.

**Tech Stack:** C++17, Eigen 3.4, Catch2 3.8.1, CMake/Ninja, OpenGL 4.1, Dear ImGui 1.91.8.

## Global Constraints

- Both 2D and 3D sessions remain paused at startup.
- Default workbench resolution is 64 cells per axis.
- The grid and boundary-guide toggles are enabled by default and affect rendering only.
- 2D draws the complete grid; 3D draws the lower X, Y, and Z face grids, not a full volume lattice.
- Overlay metadata comes from the running simulation and changes only after reset.
- Slip/sticky influence guides use `(boundary_layers + 0.5) / cells_per_axis`; outflow guides use `outflow_deletion_layers / cells_per_axis`.
- Do not change core boundary behavior, the normalized unit domain, or add sparse/world-scale grids.

---

### Task 1: Restore the default free-fall scene

**Files:**
- Modify: `src/simulation_session.h:20-37`
- Modify: `tests/simulation_session_test.cpp`

**Interfaces:**
- Consumes: existing `SimulationSessionController`, `set_active_paused`, and `advance_active` APIs.
- Produces: `SessionBlueprint<Dim>::cells_per_axis` defaulting to `64`; regression coverage for pause state and sustained free fall.

- [ ] **Step 1: Write failing default and free-fall tests**

Add to `tests/simulation_session_test.cpp`:

```cpp
TEST_CASE("default sessions start paused on a 64-cell grid", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;

    REQUIRE(sessions.active_paused());
    REQUIRE(sessions.simulation_2d().config().cells_per_axis == 64);
    REQUIRE(sessions.simulation_3d().config().cells_per_axis == 64);
}

TEST_CASE("unpaused default scene sustains free fall before boundary contact", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    sessions.set_active_paused(false);

    for (int frame = 0; frame < 50; ++frame) { sessions.advance_active(1.0e-3F); }

    float mean_vertical_velocity = 0.0F;
    for (const auto &particle : sessions.simulation_2d().particles()) {
        mean_vertical_velocity += particle.velocity(1);
    }
    mean_vertical_velocity /= static_cast<float>(sessions.simulation_2d().particles().size());

    REQUIRE(sessions.simulation_2d().simulation_time() == Approx(0.05F).margin(1.0e-5F));
    REQUIRE(mean_vertical_velocity == Approx(-0.4905F).margin(1.0e-3F));
}
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests
./build-dev/tests/nuclear_mpm_tests "default sessions start paused on a 64-cell grid,unpaused default scene sustains free fall before boundary contact" --reporter compact
```

Expected: failure because both simulations currently use 8 cells and the default boundary suppresses the expected vertical velocity.

- [ ] **Step 3: Change the workbench blueprint default**

In `src/simulation_session.h`, change only:

```cpp
int cells_per_axis = 64;
```

Leave `paused_2d_` and `paused_3d_` initialized to `true`.

- [ ] **Step 4: Run session tests and verify success**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests
./build-dev/tests/nuclear_mpm_tests "[session]" --reporter compact
```

Expected: all session tests pass, including sustained free fall.

- [ ] **Step 5: Commit the default fix**

```bash
git add src/simulation_session.h tests/simulation_session_test.cpp
git commit -m "fix: restore workbench free fall"
```

---

### Task 2: Carry active-grid metadata and expose overlay controls

**Files:**
- Modify: `src/visualizer.h:18-31`
- Modify: `src/simulation_session.cpp:132-147`
- Modify: `src/simulation_workbench.h:21-31`
- Modify: `src/simulation_workbench.cpp:152-218`
- Modify: `tests/simulation_session_test.cpp`

**Interfaces:**
- Consumes: `Simulation<Dim>::config()`, `DomainBoundaries<Dim>`, and `RenderScene`.
- Produces:
  - `RenderGridOverlay` with `cells_per_axis`, `boundary_layers`, `outflow_deletion_layers`, three-element lower/upper mode arrays, and `show_grid`/`show_boundaries` flags.
  - `RenderScene::grid_overlay`.
  - Workbench fields `show_grid_` and `show_boundaries_`, both defaulting to `true`.

- [ ] **Step 1: Write a failing active-metadata test**

Add to `tests/simulation_session_test.cpp`:

```cpp
TEST_CASE("render overlay describes the active simulation rather than staged settings", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    const auto initial = sessions.active_render_scene();

    REQUIRE(initial.grid_overlay.cells_per_axis == 64);
    REQUIRE(initial.grid_overlay.boundary_layers == 3);
    REQUIRE(initial.grid_overlay.outflow_deletion_layers == 2);
    REQUIRE(initial.grid_overlay.lower[0] == nuclear_mpm::BoundaryMode::Slip);

    sessions.staged_2d().cells_per_axis = 32;
    sessions.staged_2d().boundaries.lower[0] = nuclear_mpm::BoundaryMode::Sticky;
    const auto staged_only = sessions.active_render_scene();
    REQUIRE(staged_only.grid_overlay.cells_per_axis == 64);
    REQUIRE(staged_only.grid_overlay.lower[0] == nuclear_mpm::BoundaryMode::Slip);

    sessions.reset_active();
    const auto reset = sessions.active_render_scene();
    REQUIRE(reset.grid_overlay.cells_per_axis == 32);
    REQUIRE(reset.grid_overlay.lower[0] == nuclear_mpm::BoundaryMode::Sticky);
}
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests
./build-dev/tests/nuclear_mpm_tests "render overlay describes the active simulation rather than staged settings" --reporter compact
```

Expected: compile failure because `RenderScene::grid_overlay` does not exist.

- [ ] **Step 3: Add render metadata types**

In `src/visualizer.h`, include `<nuclear_mpm/boundary.hpp>` and `<array>`, then add:

```cpp
struct RenderGridOverlay {
    int cells_per_axis = 0;
    int boundary_layers = 0;
    int outflow_deletion_layers = 0;
    std::array<BoundaryMode, 3> lower{
            BoundaryMode::Slip, BoundaryMode::Slip, BoundaryMode::Slip};
    std::array<BoundaryMode, 3> upper{
            BoundaryMode::Slip, BoundaryMode::Slip, BoundaryMode::Slip};
    bool show_grid = true;
    bool show_boundaries = true;
};
```

Add this member to `RenderScene`:

```cpp
RenderGridOverlay grid_overlay;
```

- [ ] **Step 4: Populate metadata from the active simulation**

In `src/simulation_session.cpp`, add a local template helper:

```cpp
template<int Dim>
auto populate_grid_overlay(RenderGridOverlay &overlay, const Simulation<Dim> &simulation) -> void {
    const auto &config = simulation.config();
    overlay.cells_per_axis = config.cells_per_axis;
    overlay.boundary_layers = config.boundaries.boundary_layers;
    overlay.outflow_deletion_layers = config.boundaries.outflow_deletion_layers;
    for (int axis = 0; axis < Dim; ++axis) {
        overlay.lower[static_cast<std::size_t>(axis)] = config.boundaries.lower[axis];
        overlay.upper[static_cast<std::size_t>(axis)] = config.boundaries.upper[axis];
    }
}
```

Call it in each branch of `active_render_scene()` before copying particles:

```cpp
populate_grid_overlay(scene.grid_overlay, *simulation_2d_);
```

and:

```cpp
populate_grid_overlay(scene.grid_overlay, *simulation_3d_);
```

- [ ] **Step 5: Add workbench visualization controls**

In `src/simulation_workbench.h`, add:

```cpp
auto draw_visualization_settings() -> void;
```

and fields:

```cpp
bool show_grid_ = true;
bool show_boundaries_ = true;
```

In `src/simulation_workbench.cpp`, implement:

```cpp
auto SimulationWorkbench::draw_visualization_settings() -> void {
    ImGui::Checkbox("Show grid", &show_grid_);
    ImGui::Checkbox("Show boundary guides", &show_boundaries_);
    ImGui::TextUnformatted("Grid: blue\nSlip: teal\nSticky: red\nOutflow: amber");
}
```

Add a default-open `Visualization` collapsing header in `draw_controls()`. In `run()`, make the scene mutable and apply visibility without changing the simulation:

```cpp
auto scene = sessions_.active_render_scene();
scene.grid_overlay.show_grid = show_grid_;
scene.grid_overlay.show_boundaries = show_boundaries_;
```

- [ ] **Step 6: Run session tests and build the workbench**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests
./build-dev/tests/nuclear_mpm_tests "[session]" --reporter compact
cmake --build build --target nuclear_mpm_example
```

Expected: session tests pass and the workbench compiles; rendering is unchanged until Task 3.

- [ ] **Step 7: Commit metadata and controls**

```bash
git add src/visualizer.h src/simulation_session.cpp src/simulation_workbench.h src/simulation_workbench.cpp tests/simulation_session_test.cpp
git commit -m "feat: expose workbench overlay controls"
```

---

### Task 3: Generate and render wireframe grid and boundary guides

**Files:**
- Create: `src/overlay_geometry.h`
- Create: `src/overlay_geometry.cpp`
- Create: `tests/overlay_geometry_test.cpp`
- Modify: `src/visualizer.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `RenderScene` and `RenderGridOverlay` from Task 2.
- Produces:
  - `RenderLine { Eigen::Vector3f start, end; std::uint32_t color; }`.
  - `build_grid_lines(const RenderScene&) -> std::vector<RenderLine>`.
  - `build_boundary_guide_lines(const RenderScene&) -> std::vector<RenderLine>`.
  - OpenGL line rendering driven solely by those generated vectors.

- [ ] **Step 1: Declare overlay geometry**

Create `src/overlay_geometry.h`:

```cpp
#pragma once

#include "visualizer.h"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace nuclear_mpm::viz {

constexpr std::uint32_t kGridColor = 0x355C7DU;
constexpr std::uint32_t kSlipBoundaryColor = 0x4FB99FU;
constexpr std::uint32_t kStickyBoundaryColor = 0xE15B4FU;
constexpr std::uint32_t kOutflowBoundaryColor = 0xF2B134U;

struct RenderLine {
    Eigen::Vector3f start = Eigen::Vector3f::Zero();
    Eigen::Vector3f end = Eigen::Vector3f::Zero();
    std::uint32_t color = 0;
};

[[nodiscard]] auto build_grid_lines(const RenderScene &scene) -> std::vector<RenderLine>;
[[nodiscard]] auto build_boundary_guide_lines(const RenderScene &scene) -> std::vector<RenderLine>;

} // namespace nuclear_mpm::viz
```

Create an intentionally empty `src/overlay_geometry.cpp` so CMake can compile the new library before the functions are implemented:

```cpp
#include "overlay_geometry.h"
```

- [ ] **Step 2: Write failing geometry tests**

Create `tests/overlay_geometry_test.cpp` with:

```cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "overlay_geometry.h"

using Catch::Approx;

TEST_CASE("2D grid overlay contains every horizontal and vertical grid line", "[overlay]") {
    nuclear_mpm::viz::RenderScene scene;
    scene.dimension = nuclear_mpm::RuntimeDimension::Two;
    scene.grid_overlay.cells_per_axis = 4;

    const auto lines = nuclear_mpm::viz::build_grid_lines(scene);

    REQUIRE(lines.size() == 10);
    REQUIRE(lines.front().start.x() == Approx(0.0F));
    REQUIRE(lines.front().end.y() == Approx(1.0F));
    REQUIRE(lines.back().start.y() == Approx(1.0F));
    REQUIRE(lines.back().end.x() == Approx(1.0F));
}

TEST_CASE("3D grid overlay uses three domain faces instead of a volume lattice", "[overlay]") {
    nuclear_mpm::viz::RenderScene scene;
    scene.dimension = nuclear_mpm::RuntimeDimension::Three;
    scene.grid_overlay.cells_per_axis = 4;

    const auto lines = nuclear_mpm::viz::build_grid_lines(scene);

    REQUIRE(lines.size() == 30);
}

TEST_CASE("boundary guides use stencil influence and outflow deletion planes", "[overlay]") {
    nuclear_mpm::viz::RenderScene scene;
    scene.dimension = nuclear_mpm::RuntimeDimension::Two;
    scene.grid_overlay.cells_per_axis = 8;
    scene.grid_overlay.boundary_layers = 3;
    scene.grid_overlay.outflow_deletion_layers = 2;
    scene.grid_overlay.lower[0] = nuclear_mpm::BoundaryMode::Slip;
    scene.grid_overlay.upper[0] = nuclear_mpm::BoundaryMode::Sticky;
    scene.grid_overlay.lower[1] = nuclear_mpm::BoundaryMode::Outflow;

    const auto lines = nuclear_mpm::viz::build_boundary_guide_lines(scene);

    REQUIRE(lines.size() == 4);
    REQUIRE(lines[0].start.x() == Approx(0.4375F));
    REQUIRE(lines[0].color == nuclear_mpm::viz::kSlipBoundaryColor);
    REQUIRE(lines[1].start.x() == Approx(0.5625F));
    REQUIRE(lines[1].color == nuclear_mpm::viz::kStickyBoundaryColor);
    REQUIRE(lines[2].start.y() == Approx(0.25F));
    REQUIRE(lines[2].color == nuclear_mpm::viz::kOutflowBoundaryColor);
}

TEST_CASE("overlay visibility flags suppress generated geometry", "[overlay]") {
    nuclear_mpm::viz::RenderScene scene;
    scene.grid_overlay.cells_per_axis = 8;
    scene.grid_overlay.boundary_layers = 3;
    scene.grid_overlay.show_grid = false;
    scene.grid_overlay.show_boundaries = false;

    REQUIRE(nuclear_mpm::viz::build_grid_lines(scene).empty());
    REQUIRE(nuclear_mpm::viz::build_boundary_guide_lines(scene).empty());
}
```

- [ ] **Step 3: Register the module and test, then verify failure**

In top-level `CMakeLists.txt`, add an always-built library after `nuclear_mpm_camera`:

```cmake
add_library(nuclear_mpm_overlay src/overlay_geometry.cpp)
target_include_directories(nuclear_mpm_overlay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_overlay PUBLIC nuclear_mpm_camera)
nuclear_mpm_enable_warnings(nuclear_mpm_overlay)
nuclear_mpm_enable_sanitizers(nuclear_mpm_overlay)
```

Add `nuclear_mpm_overlay` to `nuclear_mpm_visualizer`'s private links. Add `overlay_geometry_test.cpp` and `nuclear_mpm_overlay` to `tests/CMakeLists.txt`.

Run:

```bash
cmake -S . -B build-dev -GNinja -DCMAKE_BUILD_TYPE=Debug -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
cmake --build build-dev --target nuclear_mpm_tests
```

Expected: link failure because the declared geometry functions are not implemented.

- [ ] **Step 4: Implement wire generation**

Replace the empty `src/overlay_geometry.cpp` with the implementation. Start with:

```cpp
#include "overlay_geometry.h"

#include <algorithm>
#include <array>

namespace nuclear_mpm::viz {
namespace {

[[nodiscard]] auto boundary_color(BoundaryMode mode) -> std::uint32_t {
    switch (mode) {
        case BoundaryMode::Slip: return kSlipBoundaryColor;
        case BoundaryMode::Sticky: return kStickyBoundaryColor;
        case BoundaryMode::Outflow: return kOutflowBoundaryColor;
    }
    return kSlipBoundaryColor;
}

[[nodiscard]] auto guide_fraction(const RenderGridOverlay &overlay, BoundaryMode mode) -> float {
    const float layers = mode == BoundaryMode::Outflow
            ? static_cast<float>(overlay.outflow_deletion_layers)
            : static_cast<float>(overlay.boundary_layers) + 0.5F;
    return layers / static_cast<float>(overlay.cells_per_axis);
}

} // namespace
```

For `build_grid_lines`:

- Return empty when hidden or `cells_per_axis <= 0`.
- Map each `i / cells_per_axis` through `domain_min + fraction * (domain_max - domain_min)`.
- Emit `2 * (N + 1)` lines for 2D.
- Emit the same two-axis grid on each lower X, lower Y, and lower Z face for `6 * (N + 1)` lines in 3D.
- Set every line color to `kGridColor`.

For `build_boundary_guide_lines`:

- Return empty when hidden or `cells_per_axis <= 0`.
- Iterate active dimensions and lower/upper faces in axis order.
- Use `guide_fraction` and the matching face's mode.
- Emit one full-domain line per 2D face.
- Emit a four-edge rectangle per 3D face.
- Set each line's color from `boundary_color`.

- [ ] **Step 5: Run overlay tests and verify success**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests
./build-dev/tests/nuclear_mpm_tests "[overlay]" --reporter compact
```

Expected: all overlay geometry tests pass.

- [ ] **Step 6: Render generated lines before particles**

In `src/visualizer.cpp`:

- Include `overlay_geometry.h`.
- Add a helper that converts every `RenderLine` endpoint through the existing `to_vertex` function.
- In `draw_scene`, after drawing the existing domain boundary and before drawing particles, concatenate `build_grid_lines(scene)` and `build_boundary_guide_lines(scene)`.
- Upload the resulting vertices with `glBufferData` and draw with `glDrawArrays(GL_LINES, ...)`.
- Use line width `1.0F` for the shared overlay draw; colors remain encoded per vertex.

- [ ] **Step 7: Run complete verification**

Run:

```bash
just test
just build-workbench
```

Expected: all tests pass and `nuclear_mpm_example` builds.

Manually run:

```bash
./build/nuclear_mpm_example
```

Verify that it starts paused, reports 64 cells, shows the 2D grid and boundary guides, renders three face grids in 3D, toggles each overlay independently, and free-falls after unpausing.

- [ ] **Step 8: Commit overlay rendering**

```bash
git add CMakeLists.txt tests/CMakeLists.txt src/overlay_geometry.h src/overlay_geometry.cpp src/visualizer.cpp tests/overlay_geometry_test.cpp
git commit -m "feat: visualize MPM grid boundaries"
```
