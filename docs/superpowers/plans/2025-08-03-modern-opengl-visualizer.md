# Modern OpenGL Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bundled Taichi GUI with a shader-based OpenGL 4.1 particle visualizer and make the complete project build through Just with CMake-fetched dependencies.

**Architecture:** A focused `nuclear_mpm_visualizer` library hides GLFW, GLAD, shaders, and GPU resources behind a Pimpl C++ API. CMake fetches Eigen and GLFW, while a reproducibly generated GLAD loader is committed under `vendor/glad`; the example and optional solver visualization convert simulation particles into renderer-neutral points.

**Tech Stack:** C++17, OpenGL 4.1 Core, GLSL 410 Core, GLFW 3.4, GLAD 2.0.8, Eigen 3.4.0, CMake FetchContent, Ninja, Just

## Global Constraints

- Request OpenGL 4.1 Core and use GLSL 410 Core shaders.
- Fetch pinned Eigen 3.4.0 and GLFW 3.4 through CMake.
- Build the checked-in loader generated reproducibly by GLAD 2.0.8 for `gl:core=4.1` with no extensions.
- Raise the CMake minimum to exactly 3.16.
- Keep `flags` as a Git submodule and initialize it in `just build`.
- Remove direct X11 discovery and linking.
- Delete `src/taichi.h` and all Taichi references.
- Configure from inside `build/` with `cmake -GNinja ..` and compile with `ninja`.
- Keep `just run` targeting `./build/nuclear_mpm_example`.

---

### Task 1: Add the OpenGL renderer library and fetched graphics dependencies

**Files:**
- Create: `src/visualizer.h`
- Create: `src/visualizer.cpp`
- Create: `vendor/glad/include/glad/gl.h`
- Create: `vendor/glad/include/KHR/khrplatform.h`
- Create: `vendor/glad/src/gl.c`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: GLFW window/context APIs, GLAD OpenGL 4.1 entry points, and normalized simulation coordinates.
- Produces: `nclr::viz::Point`, `nclr::viz::Visualizer`, and the `nuclear_mpm_visualizer` CMake target.

- [ ] **Step 1: Preserve the observed failing build evidence**

Run:

```bash
just build
```

Expected: FAIL compiling `src/taichi.h` on Apple Silicon because it includes x86 intrinsics and `rdtsc` assembly.

- [ ] **Step 2: Add the renderer-independent public header**

Create `src/visualizer.h`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nclr::viz {

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

} // namespace nclr::viz
```

- [ ] **Step 3: Implement the OpenGL 4.1 renderer**

Create `src/visualizer.cpp` with:

```cpp
#include "visualizer.h"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nclr::viz {
namespace {

constexpr auto kVertexShader = R"(
#version 410 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
out vec3 vertex_color;
uniform float point_size;

void main() {
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = point_size;
    vertex_color = color;
}
)";

constexpr auto kFragmentShader = R"(
#version 410 core
in vec3 vertex_color;
out vec4 fragment_color;

void main() {
    fragment_color = vec4(vertex_color, 1.0);
}
)";

struct Vertex {
    float x;
    float y;
    float r;
    float g;
    float b;
};

auto to_vertex(float x, float y, std::uint32_t color) -> Vertex {
    constexpr auto channel_scale = 1.0F / 255.0F;
    return {
            x,
            y,
            static_cast<float>((color >> 16U) & 0xFFU) * channel_scale,
            static_cast<float>((color >> 8U) & 0xFFU) * channel_scale,
            static_cast<float>(color & 0xFFU) * channel_scale,
    };
}

auto compile_shader(GLenum type, const char *source) -> GLuint {
    const auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) { return shader; }

    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    auto log = std::string(static_cast<std::size_t>(log_length), '\0');
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    glDeleteShader(shader);
    throw std::runtime_error("OpenGL shader compilation failed: " + log);
}

auto link_program(GLuint vertex_shader, GLuint fragment_shader) -> GLuint {
    const auto program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) { return program; }

    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    auto log = std::string(static_cast<std::size_t>(log_length), '\0');
    glGetProgramInfoLog(program, log_length, nullptr, log.data());
    glDeleteProgram(program);
    throw std::runtime_error("OpenGL program linking failed: " + log);
}

} // namespace

struct Visualizer::Impl {
    GLFWwindow *window = nullptr;
    GLuint program = 0;
    GLuint vertex_array = 0;
    GLuint vertex_buffer = 0;

    Impl(const std::string &title, int width, int height) {
        if (glfwInit() != GLFW_TRUE) { throw std::runtime_error("GLFW initialization failed"); }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

        window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            glfwTerminate();
            throw std::runtime_error("GLFW window creation failed");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("GLAD OpenGL loading failed");
        }

        try {
            const auto vertex_shader = compile_shader(GL_VERTEX_SHADER, kVertexShader);
            GLuint fragment_shader = 0;
            try {
                fragment_shader = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader);
                program = link_program(vertex_shader, fragment_shader);
            } catch (...) {
                glDeleteShader(vertex_shader);
                if (fragment_shader != 0) { glDeleteShader(fragment_shader); }
                throw;
            }
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);

            glGenVertexArrays(1, &vertex_array);
            glGenBuffers(1, &vertex_buffer);
            glBindVertexArray(vertex_array);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, x)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(offsetof(Vertex, r)));
            glEnable(GL_PROGRAM_POINT_SIZE);
        } catch (...) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw;
        }
    }

    ~Impl() {
        if (window != nullptr) { glfwMakeContextCurrent(window); }
        if (vertex_buffer != 0) { glDeleteBuffers(1, &vertex_buffer); }
        if (vertex_array != 0) { glDeleteVertexArrays(1, &vertex_array); }
        if (program != 0) { glDeleteProgram(program); }
        if (window != nullptr) { glfwDestroyWindow(window); }
        glfwTerminate();
    }

    auto draw(const std::vector<Point> &points) -> void {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { glfwSetWindowShouldClose(window, GLFW_TRUE); }

        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        glViewport(0, 0, framebuffer_width, framebuffer_height);
        glClearColor(0x11 / 255.0F, 0x2F / 255.0F, 0x41 / 255.0F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glUniform1f(glGetUniformLocation(program, "point_size"), 4.0F);
        glBindVertexArray(vertex_array);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

        const auto boundary = std::array{
                to_vertex(0.04F, 0.04F, 0x4FB99FU),
                to_vertex(0.96F, 0.04F, 0x4FB99FU),
                to_vertex(0.96F, 0.96F, 0x4FB99FU),
                to_vertex(0.04F, 0.96F, 0x4FB99FU),
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(boundary), boundary.data(), GL_DYNAMIC_DRAW);
        glLineWidth(2.0F);
        glDrawArrays(GL_LINE_LOOP, 0, static_cast<GLsizei>(boundary.size()));

        auto vertices = std::vector<Vertex>{};
        vertices.reserve(points.size());
        for (const auto &point : points) { vertices.push_back(to_vertex(point.x, point.y, point.color)); }
        if (!vertices.empty()) {
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(),
                         GL_DYNAMIC_DRAW);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
};

Visualizer::Visualizer(std::string title, int width, int height)
    : impl_(std::make_unique<Impl>(std::move(title), width, height)) {}

Visualizer::~Visualizer() = default;

auto Visualizer::should_close() const -> bool { return glfwWindowShouldClose(impl_->window) == GLFW_TRUE; }

auto Visualizer::draw(const std::vector<Point> &points) -> void { impl_->draw(points); }

} // namespace nclr::viz
```

- [ ] **Step 4: Replace CMake dependency and target definitions**

Update `CMakeLists.txt` to use CMake 3.16 and fetch Eigen 3.4.0 and GLFW 3.4 with `FetchContent_MakeAvailable`. Build the generated loader directly:

```cmake
add_library(glad_gl_core_41 STATIC vendor/glad/src/gl.c)
target_include_directories(glad_gl_core_41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad/include)
find_package(OpenGL REQUIRED)
```

Disable GLFW docs, tests, examples, and installation through forced cache options. Add `nuclear_mpm_visualizer` from `src/visualizer.cpp`, expose `src/`, and link it privately to `glfw`, `glad_gl_core_41`, and `OpenGL::GL`. Remove `find_package(X11 REQUIRED)` and all `${X11_LIBRARIES}` links.

- [ ] **Step 5: Build the renderer target**

Run:

```bash
rm -rf build
mkdir -p build
cd build && cmake -GNinja ..
cd build && ninja nuclear_mpm_visualizer
```

Expected: CMake fetches Eigen and GLFW, compiles the bundled OpenGL 4.1 loader, and builds `nuclear_mpm_visualizer` successfully.

---

### Task 2: Migrate the example to the new renderer

**Files:**
- Modify: `src/example.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `nclr::viz::Visualizer`, `nclr::viz::Point`, simulation particles, and the existing 3D projection helper.
- Produces: A GUI example that advances until its GLFW window closes and renders through OpenGL 4.1.

- [ ] **Step 1: Verify the example still fails through Taichi**

Run:

```bash
cmake --build build --target nuclear_mpm_example
```

Expected: FAIL because `src/example.cpp` still includes `src/taichi.h` and compiles x86-only intrinsics on Apple Silicon.

- [ ] **Step 2: Replace Taichi calls in the example**

Replace the Taichi include with `#include "visualizer.h"`. Construct `nclr::viz::Visualizer visualizer("Nuclear MPM", kWindowSize, kWindowSize);`. Change the infinite loop to stop when `visualizer.should_close()`. At each existing frame interval, build a `std::vector<nclr::viz::Point>`, preserving the existing 3D projection or using `p.x(0)` and `p.x(1)` in 2D, and call `visualizer.draw(points)`.

- [ ] **Step 3: Link and build the example**

Link `nuclear_mpm_example` to `nuclear_mpm_visualizer` and `Eigen3::Eigen`, then run:

```bash
cmake --build build --target nuclear_mpm_example
test -x build/nuclear_mpm_example
```

Expected: PASS and produce the GUI executable without compiling Taichi.

---

### Task 3: Migrate optional solver visualization and remove Taichi

**Files:**
- Modify: `src/solver.cpp`
- Modify: `CMakeLists.txt`
- Delete: `src/taichi.h`

**Interfaces:**
- Consumes: Saved two-dimensional solver states and the shared visualizer API.
- Produces: Optional shader-based solver playback and a source tree with no Taichi dependency.

- [ ] **Step 1: Verify optional solver visualization still references Taichi**

Run:

```bash
rg -n 'taichi\.h|taichi::' src/solver.cpp src/taichi.h
```

Expected: PASS with matches, proving the obsolete implementation remains.

- [ ] **Step 2: Replace solver Taichi calls**

Under `NCLR_SOLVER_VIZ`, include `visualizer.h`, create `nclr::viz::Visualizer visualizer("Results", kWindowSize, kWindowSize)`, stop playback if the window closes, convert each state to `std::vector<nclr::viz::Point>`, and call `visualizer.draw(points)`.

- [ ] **Step 3: Link optional solver visualization and delete Taichi**

When `WITH_NCLR_SOLVER_VIZ` is enabled, link `nuclear_mpm_solver` to `nuclear_mpm_visualizer`. Delete `src/taichi.h`.

- [ ] **Step 4: Build normal and visualized solver configurations**

Run:

```bash
cmake --build build --target nuclear_mpm_solver
rm -rf build-viz
cmake -S . -B build-viz -GNinja -DWITH_NCLR_SOLVER_VIZ=ON
cmake --build build-viz --target nuclear_mpm_solver
./build/nuclear_mpm_solver --help
```

Expected: Both solver variants build. The help command prints usage without a GUI.

---

### Task 4: Complete the Just workflow, documentation, and verification

**Files:**
- Modify: `justfile`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2025-08-03-justfile-build-run-design.md`
- Modify: `docs/superpowers/plans/2025-08-03-justfile-build-run.md`
- Create: `docs/superpowers/plans/2025-08-03-modern-opengl-visualizer.md`

**Interfaces:**
- Consumes: Git submodules and completed CMake targets.
- Produces: A fresh-clone `just build`, GUI `just run`, `just clean`, and accurate project documentation.

- [ ] **Step 1: Verify Just does not initialize submodules**

Run:

```bash
just --dry-run build | grep -F 'git submodule update --init --recursive'
```

Expected: FAIL because the current build recipe omits submodule initialization.

- [ ] **Step 2: Add submodule initialization to Just**

Add `git submodule update --init --recursive` as the first command in `build`; preserve the approved CMake/Ninja commands and existing `run` and `clean` recipes.

- [ ] **Step 3: Update the README**

Remove the embedded Taichi sample and all Taichi references. Describe `src/visualizer.h` as the optional example renderer, list CMake/Ninja and platform OpenGL development libraries as requirements, state that CMake fetches Eigen/GLFW and builds bundled generated GLAD, and document `just build`, `just run`, and `just clean`.

- [ ] **Step 4: Verify the complete workflow**

Run:

```bash
rm -rf build
just build
test -x build/nuclear_mpm_example
test -x build/nuclear_mpm_solver
just build
ctest --test-dir build --output-on-failure
just --dry-run run
just --dry-run clean
rg -n 'taichi\.h|taichi::|Taichi' src CMakeLists.txt README.md
```

Expected: Both clean and incremental builds pass; the 3D compile check, solver help, and one-step solver tests pass; run and clean resolve correctly; the final `rg` exits 1 with no matches.

- [ ] **Step 5: Review and commit**

Run `git diff --check`, inspect `git diff --stat` and `git status --short`, then commit:

```bash
git add CMakeLists.txt README.md justfile src/visualizer.h src/visualizer.cpp src/example.cpp src/solver.cpp src/taichi.h docs/superpowers

git commit -m "build: replace Taichi with modern OpenGL visualizer"
```
