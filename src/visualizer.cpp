#include "visualizer.h"

#include "overlay_geometry.h"

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuclear_mpm::viz {
namespace {

constexpr auto kVertexShader = R"(
#version 410 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
out vec3 vertex_color;
uniform mat4 view_projection;
uniform float point_size;

void main() {
    gl_Position = view_projection * vec4(position, 1.0);
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
    float z;
    float r;
    float g;
    float b;
};

[[nodiscard]] auto to_vertex(float x, float y, float z, std::uint32_t color) -> Vertex {
    constexpr auto channel_scale = 1.0F / 255.0F;
    return {
            x,
            y,
            z,
            static_cast<float>((color >> 16U) & 0xFFU) * channel_scale,
            static_cast<float>((color >> 8U) & 0xFFU) * channel_scale,
            static_cast<float>(color & 0xFFU) * channel_scale,
    };
}

[[nodiscard]] auto line_vertices(const std::vector<RenderLine> &lines) -> std::vector<Vertex> {
    std::vector<Vertex> vertices;
    vertices.reserve(lines.size() * 2);
    for (const auto &line : lines) {
        vertices.push_back(to_vertex(line.start.x(), line.start.y(), line.start.z(), line.color));
        vertices.push_back(to_vertex(line.end.x(), line.end.y(), line.end.z(), line.color));
    }
    return vertices;
}

[[nodiscard]] auto compile_shader(GLenum type, const char *source) -> GLuint {
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

[[nodiscard]] auto link_program(GLuint vertex_shader, GLuint fragment_shader) -> GLuint {
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

[[nodiscard]] auto boundary_vertices(const RenderScene &scene) -> std::vector<Vertex> {
    constexpr std::uint32_t color = 0x4FB99FU;
    const auto &minimum = scene.domain_min;
    const auto &maximum = scene.domain_max;
    if (scene.dimension == RuntimeDimension::Two) {
        return {
                to_vertex(minimum.x(), minimum.y(), 0.0F, color),
                to_vertex(maximum.x(), minimum.y(), 0.0F, color),
                to_vertex(maximum.x(), maximum.y(), 0.0F, color),
                to_vertex(minimum.x(), maximum.y(), 0.0F, color),
        };
    }

    const std::array<Eigen::Vector3f, 8> corners{
            Eigen::Vector3f{minimum.x(), minimum.y(), minimum.z()},
            Eigen::Vector3f{maximum.x(), minimum.y(), minimum.z()},
            Eigen::Vector3f{maximum.x(), maximum.y(), minimum.z()},
            Eigen::Vector3f{minimum.x(), maximum.y(), minimum.z()},
            Eigen::Vector3f{minimum.x(), minimum.y(), maximum.z()},
            Eigen::Vector3f{maximum.x(), minimum.y(), maximum.z()},
            Eigen::Vector3f{maximum.x(), maximum.y(), maximum.z()},
            Eigen::Vector3f{minimum.x(), maximum.y(), maximum.z()},
    };
    constexpr std::array<std::array<int, 2>, 12> edges{
            std::array<int, 2>{0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    std::vector<Vertex> vertices;
    vertices.reserve(edges.size() * 2);
    for (const auto &edge : edges) {
        for (const int corner_index : edge) {
            const auto &corner = corners[static_cast<std::size_t>(corner_index)];
            vertices.push_back(to_vertex(corner.x(), corner.y(), corner.z(), color));
        }
    }
    return vertices;
}

} // namespace

struct Visualizer::Impl {
    GLFWwindow *window = nullptr;
    GLuint program = 0;
    GLuint vertex_array = 0;
    GLuint vertex_buffer = 0;
    CameraInput camera_input_value;
    double previous_cursor_x = 0.0;
    double previous_cursor_y = 0.0;
    double scroll_delta = 0.0;
    bool cursor_initialized = false;

    Impl(const std::string &title, int width, int height) {
        if (width <= 0 || height <= 0) { throw std::invalid_argument("visualizer dimensions must be positive"); }
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
        glfwSetWindowUserPointer(window, this);
        glfwSetScrollCallback(window, [](GLFWwindow *callback_window, double, double y_offset) {
            auto *implementation = static_cast<Impl *>(glfwGetWindowUserPointer(callback_window));
            if (implementation != nullptr) { implementation->scroll_delta += y_offset; }
        });

        if (gladLoadGL(reinterpret_cast<GLADloadfunc>(glfwGetProcAddress)) == 0) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("GLAD OpenGL loading failed");
        }

        try {
            const auto vertex_shader = compile_shader(GL_VERTEX_SHADER, kVertexShader);
            const auto fragment_shader = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader);
            try {
                program = link_program(vertex_shader, fragment_shader);
            } catch (...) {
                glDeleteShader(vertex_shader);
                glDeleteShader(fragment_shader);
                throw;
            }
            glDeleteShader(vertex_shader);
            glDeleteShader(fragment_shader);

            glGenVertexArrays(1, &vertex_array);
            glGenBuffers(1, &vertex_buffer);
            glBindVertexArray(vertex_array);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<void *>(offsetof(Vertex, x)));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                                  reinterpret_cast<void *>(offsetof(Vertex, r)));
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

    auto begin_frame() -> void {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) { glfwSetWindowShouldClose(window, GLFW_TRUE); }

        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);
        Eigen::Vector2f cursor_delta = Eigen::Vector2f::Zero();
        const bool orbit_active = cursor_initialized &&
                                  glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool pan_active = cursor_initialized &&
                                glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (cursor_initialized) {
            cursor_delta.x() = static_cast<float>(cursor_x - previous_cursor_x);
            cursor_delta.y() = static_cast<float>(cursor_y - previous_cursor_y);
        }
        camera_input_value = map_pointer_camera_input(cursor_delta, orbit_active, pan_active,
                                                      static_cast<float>(scroll_delta));
        cursor_initialized = true;
        previous_cursor_x = cursor_x;
        previous_cursor_y = cursor_y;
        scroll_delta = 0.0;

        int framebuffer_width = 0;
        int framebuffer_height = 0;
        glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
        glViewport(0, 0, framebuffer_width, framebuffer_height);
        glClearColor(0x11 / 255.0F, 0x2F / 255.0F, 0x41 / 255.0F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    auto draw_scene(const RenderScene &scene, const Eigen::Matrix4f &view_projection) -> void {
        if (!scene.domain_min.allFinite() || !scene.domain_max.allFinite() || !view_projection.allFinite()) {
            throw std::invalid_argument("render scene and view-projection matrix must be finite");
        }
        if (scene.dimension == RuntimeDimension::Three) {
            glEnable(GL_DEPTH_TEST);
        } else {
            glDisable(GL_DEPTH_TEST);
        }

        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "view_projection"), 1, GL_FALSE, view_projection.data());
        glUniform1f(glGetUniformLocation(program, "point_size"), scene.point_size);
        glBindVertexArray(vertex_array);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

        const auto boundary = boundary_vertices(scene);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(boundary.size() * sizeof(Vertex)), boundary.data(),
                     GL_DYNAMIC_DRAW);
        glLineWidth(2.0F);
        glDrawArrays(scene.dimension == RuntimeDimension::Two ? GL_LINE_LOOP : GL_LINES, 0,
                     static_cast<GLsizei>(boundary.size()));

        auto overlay_lines = build_grid_lines(scene);
        const auto boundary_guides = build_boundary_guide_lines(scene);
        overlay_lines.insert(overlay_lines.end(), boundary_guides.begin(), boundary_guides.end());
        const auto overlay = line_vertices(overlay_lines);
        if (!overlay.empty()) {
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(overlay.size() * sizeof(Vertex)), overlay.data(),
                         GL_DYNAMIC_DRAW);
            glLineWidth(1.0F);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(overlay.size()));
        }

        std::vector<Vertex> vertices;
        vertices.reserve(scene.points.size());
        for (const auto &point : scene.points) {
            vertices.push_back(to_vertex(point.x, point.y, point.z, point.color));
        }
        if (!vertices.empty()) {
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(),
                         GL_DYNAMIC_DRAW);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices.size()));
        }
    }

    auto end_frame() -> void { glfwSwapBuffers(window); }
};

Visualizer::Visualizer(std::string title, int width, int height)
    : impl_(std::make_unique<Impl>(std::move(title), width, height)) {}
Visualizer::~Visualizer() = default;

auto Visualizer::should_close() const -> bool { return glfwWindowShouldClose(impl_->window) == GLFW_TRUE; }

auto Visualizer::framebuffer_aspect_ratio() const -> float {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(impl_->window, &width, &height);
    return height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0F;
}

auto Visualizer::camera_input() const -> CameraInput { return impl_->camera_input_value; }
auto Visualizer::native_window() const noexcept -> GLFWwindow * { return impl_->window; }
auto Visualizer::begin_frame() -> void { impl_->begin_frame(); }
auto Visualizer::draw_scene(const RenderScene &scene, const Eigen::Matrix4f &view_projection) -> void {
    impl_->draw_scene(scene, view_projection);
}
auto Visualizer::end_frame() -> void { impl_->end_frame(); }

} // namespace nuclear_mpm::viz
