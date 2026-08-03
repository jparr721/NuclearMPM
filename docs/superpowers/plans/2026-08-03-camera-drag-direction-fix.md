# Camera Drag Direction Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make orbit and pan dragging move the visible workbench scene in the same screen direction as the cursor.

**Architecture:** Move GLFW pointer-delta conversion into a pure helper in the existing camera module, then have the visualizer delegate to it. Keep camera mathematics and input sensitivities unchanged; only correct the orbit signs at the input boundary.

**Tech Stack:** C++17, Eigen, GLFW, Catch2, CMake/Ninja

## Global Constraints

- Do not launch the interactive workbench during verification.
- Keep existing mouse assignments and sensitivity constants unchanged.
- Preserve current pan and scroll behavior.
- Use a headless regression test and run only bounded suites.

---

### Task 1: Correct Pointer-to-Camera Mapping

**Files:**
- Modify: `src/camera.h`
- Modify: `src/camera.cpp`
- Modify: `src/visualizer.cpp`
- Test: `tests/camera_test.cpp`

**Interfaces:**
- Consumes: `CameraInput` and GLFW cursor/button/scroll state.
- Produces: `map_pointer_camera_input(const Eigen::Vector2f &cursor_delta, bool orbit_active, bool pan_active, float scroll_delta) noexcept -> CameraInput`.

- [ ] **Step 1: Declare the mapping helper and write failing direction tests**

Add to `src/camera.h`:

```cpp
[[nodiscard]] auto map_pointer_camera_input(const Eigen::Vector2f &cursor_delta, bool orbit_active,
                                            bool pan_active, float scroll_delta) noexcept -> CameraInput;
```

Add headless Catch2 tests to `tests/camera_test.cpp`:

```cpp
TEST_CASE("pointer orbit drag makes the scene follow the cursor", "[camera]") {
    const auto input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 10.0F}, true, false, 0.0F);

    REQUIRE(input.orbit_x == Approx(0.1F));
    REQUIRE(input.orbit_y == Approx(0.05F));
    REQUIRE(input.pan_x == 0.0F);
    REQUIRE(input.pan_y == 0.0F);
}

TEST_CASE("pointer pan and scroll preserve grab-style mapping", "[camera]") {
    const auto input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 10.0F}, false, true, 2.0F);

    REQUIRE(input.pan_x == Approx(-0.02F));
    REQUIRE(input.pan_y == Approx(0.01F));
    REQUIRE(input.dolly == Approx(0.3F));
    REQUIRE(input.orbit_x == 0.0F);
    REQUIRE(input.orbit_y == 0.0F);
}
```

Also add headless projection-level tests that compose the mapping with `OrbitCamera`: a right/down orbit drag must move representative projected features right/down, and a right/down pan drag must move the projected scene center right/down. Convert clip-space Y to screen direction by asserting that downward motion decreases normalized-device Y.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests --parallel 2
```

Expected: link failure because `map_pointer_camera_input` is declared but not implemented.

- [ ] **Step 3: Implement the pure mapping helper**

Add to `src/camera.cpp`:

```cpp
auto map_pointer_camera_input(const Eigen::Vector2f &cursor_delta, bool orbit_active,
                              bool pan_active, float scroll_delta) noexcept -> CameraInput {
    CameraInput input;
    if (orbit_active) {
        input.orbit_x = cursor_delta.x() * 0.005F;
        input.orbit_y = cursor_delta.y() * 0.005F;
    }
    if (pan_active) {
        input.pan_x = -cursor_delta.x() * 0.001F;
        input.pan_y = cursor_delta.y() * 0.001F;
    }
    input.dolly = scroll_delta * 0.15F;
    return input;
}
```

In `Visualizer::Impl::begin_frame()`, replace the inline sign/scaling calculations with one call to `map_pointer_camera_input`, passing the current cursor delta, left-button state, middle-button state, and accumulated scroll. Preserve the existing first-frame cursor initialization behavior and reset `scroll_delta` after conversion.

- [ ] **Step 4: Verify GREEN and run bounded regression suites**

Run:

```bash
cmake --build build-dev --target nuclear_mpm_tests nuclear_mpm_example --parallel 2
./build-dev/tests/nuclear_mpm_tests "[camera]"
ctest --test-dir build-dev --output-on-failure
```

Expected: camera tests and all bounded tests pass with no compiler warnings. Do not launch `nuclear_mpm_example`.

- [ ] **Step 5: Commit**

```bash
git add src/camera.h src/camera.cpp src/visualizer.cpp tests/camera_test.cpp \
  docs/superpowers/plans/2026-08-03-camera-drag-direction-fix.md
git commit -m "fix: correct camera drag directions"
```
