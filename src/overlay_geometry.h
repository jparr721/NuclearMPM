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
