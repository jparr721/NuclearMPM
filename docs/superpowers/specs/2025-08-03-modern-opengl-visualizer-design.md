# Modern OpenGL Visualizer Design

## Goal

Replace the bundled 30,000-line Taichi GUI header with a focused particle visualizer using OpenGL 4.1 Core, GLFW, and GLSL shaders. The example remains runnable through `just run`, and optional solver visualization uses the same renderer.

## Platform and Dependencies

- Request an OpenGL 4.1 Core context, the newest OpenGL version supported by macOS.
- Use GLSL 410 Core vertex and fragment shaders.
- Fetch pinned GLFW 3.4 and Eigen 3.4.0 sources with CMake `FetchContent`.
- Store the reproducibly generated GLAD 2.0.8 `gl:core=4.1` loader under `vendor/glad`, avoiding a build-time Python/Jinja2 dependency.
- Keep `flags` as a Git submodule initialized by `just build`.
- Remove direct X11 discovery and linking; GLFW owns platform-specific window integration.

## Components

### `src/visualizer.h`

Expose a small renderer-independent API in `nclr::viz`:

```cpp
struct Point {
    float x;
    float y;
    std::uint32_t color;
};

class Visualizer {
  public:
    Visualizer(std::string title, int width, int height);
    ~Visualizer();

    Visualizer(const Visualizer &) = delete;
    auto operator=(const Visualizer &) -> Visualizer & = delete;

    [[nodiscard]] auto should_close() const -> bool;
    auto draw(const std::vector<Point> &points) -> void;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

Use a private implementation object so GLFW and OpenGL types do not leak into simulation code.

### `src/visualizer.cpp`

Own GLFW initialization, the window and OpenGL context, GLAD loading, shader compilation/linking, and VAO/VBO cleanup. Maintain one dynamic VBO and reusable VAO. Convert packed `0xRRGGBB` colors to normalized vertex colors on the CPU.

The vertex shader maps simulation coordinates from `[0, 1]` into OpenGL normalized device coordinates `[-1, 1]`, forwards per-vertex color, and sets particle point size to 4 pixels. The fragment shader outputs the interpolated color. The boundary uses a 2-pixel line width.

Each frame:

1. Poll input and mark the window for closing when Escape is pressed.
2. Clear with `0x112F41`.
3. Upload and draw the fixed `0.04`–`0.96` boundary as a green `GL_LINE_LOOP` using `0x4FB99F`.
4. Upload particle vertices and draw them with `GL_POINTS`.
5. Swap buffers and poll GLFW events.

### Simulation call sites

`src/example.cpp` converts each simulation particle into an `nclr::viz::Point`, advances while the window remains open, and draws at the existing frame interval. Three-dimensional particles retain the existing projection and scale behavior.

When `WITH_NCLR_SOLVER_VIZ=ON`, `src/solver.cpp` converts saved two-dimensional particle states to the same point type and renders them through the shared visualizer.

## Build Structure

CMake creates a static `glad_gl_core_41` target from the generated source under `vendor/glad`. It creates a `nuclear_mpm_visualizer` library from `src/visualizer.cpp`, links it privately to GLFW, GLAD, and `OpenGL::GL`, and exposes `src/visualizer.h` to consumers. `nuclear_mpm_example` always links the visualizer. `nuclear_mpm_solver` links it only when solver visualization is enabled.

Raise the CMake minimum to 3.16 for modern `FetchContent`. GLFW examples, tests, and documentation remain disabled.

## Removal and Documentation

Delete `src/taichi.h` and remove every Taichi include and API reference. Update the README example and dependency/build notes to describe the GLFW/OpenGL renderer, CMake-fetched Eigen/GLFW dependencies, and bundled generated GLAD loader.

## Error Handling

Visualizer construction throws `std::runtime_error` when GLFW initialization, window creation, GLAD loading, shader compilation, or program linking fails. Shader/program errors include the OpenGL info log. Destruction releases OpenGL resources, destroys the window, and terminates GLFW. An empty particle list still draws the boundary and swaps the frame.

## Verification

- Configure from a clean `build/` directory without a system Eigen installation.
- Confirm CMake fetches Eigen and GLFW, compiles the bundled GLAD loader, and initializes `flags` through `just build`.
- Build both executables on Apple Silicon without compiling `taichi.h`.
- Compile-check the example in both two- and three-dimensional modes.
- Run solver help and a one-step simulation as non-GUI smoke tests through CTest.
- Launch the example long enough to create its OpenGL 4.1 window and render frames, then close it.
- Run `just build` again to verify incremental behavior.
- Confirm no Taichi references remain in source, CMake, or README.
