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
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
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
