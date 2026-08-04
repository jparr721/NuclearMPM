#pragma once

#include "camera.h"

#include <nuclear_mpm/boundary.hpp>
#include <nuclear_mpm/types.hpp>

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

namespace nuclear_mpm::viz {

struct RenderPoint {
    float x;
    float y;
    float z;
    std::uint32_t color;
};

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

struct RenderScene {
    std::vector<RenderPoint> points;
    RenderGridOverlay grid_overlay;
    RuntimeDimension dimension = RuntimeDimension::Two;
    Eigen::Vector3f domain_min = Eigen::Vector3f::Zero();
    Eigen::Vector3f domain_max = Eigen::Vector3f::Ones();
    float point_size = 4.0F;
};

class Visualizer {
  public:
    Visualizer(std::string title, int width, int height);
    ~Visualizer();

    Visualizer(const Visualizer &) = delete;
    auto operator=(const Visualizer &) -> Visualizer & = delete;

    [[nodiscard]] auto should_close() const -> bool;
    [[nodiscard]] auto framebuffer_aspect_ratio() const -> float;
    [[nodiscard]] auto camera_input() const -> CameraInput;
    [[nodiscard]] auto native_window() const noexcept -> GLFWwindow *;

    auto begin_frame() -> void;
    auto draw_scene(const RenderScene &scene, const Eigen::Matrix4f &view_projection) -> void;
    auto end_frame() -> void;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nuclear_mpm::viz
