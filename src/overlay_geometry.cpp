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

[[nodiscard]] auto coordinate_at_fraction(const RenderScene &scene, int axis, float fraction) -> float {
    return scene.domain_min[axis] + fraction * (scene.domain_max[axis] - scene.domain_min[axis]);
}

auto append_grid_face(std::vector<RenderLine> &lines, const RenderScene &scene,
                      int fixed_axis, int first_axis, int second_axis) -> void {
    const auto cells = scene.grid_overlay.cells_per_axis;
    for (int i = 0; i <= cells; ++i) {
        const auto fraction = static_cast<float>(i) / static_cast<float>(cells);
        const auto first_coordinate = coordinate_at_fraction(scene, first_axis, fraction);
        const auto second_coordinate = coordinate_at_fraction(scene, second_axis, fraction);

        auto first_start = scene.domain_min;
        auto first_end = scene.domain_min;
        first_start[fixed_axis] = scene.domain_min[fixed_axis];
        first_end[fixed_axis] = scene.domain_min[fixed_axis];
        first_start[first_axis] = first_coordinate;
        first_end[first_axis] = first_coordinate;
        first_end[second_axis] = scene.domain_max[second_axis];
        lines.push_back({first_start, first_end, kGridColor});

        auto second_start = scene.domain_min;
        auto second_end = scene.domain_min;
        second_start[fixed_axis] = scene.domain_min[fixed_axis];
        second_end[fixed_axis] = scene.domain_min[fixed_axis];
        second_start[second_axis] = second_coordinate;
        second_end[second_axis] = second_coordinate;
        second_end[first_axis] = scene.domain_max[first_axis];
        lines.push_back({second_start, second_end, kGridColor});
    }
}

auto append_boundary_rectangle(std::vector<RenderLine> &lines, const RenderScene &scene,
                               int fixed_axis, float coordinate, std::uint32_t color) -> void {
    std::array<int, 2> varying_axes{};
    int varying_index = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != fixed_axis) {
            varying_axes[static_cast<std::size_t>(varying_index)] = axis;
            ++varying_index;
        }
    }

    auto corner0 = scene.domain_min;
    auto corner1 = scene.domain_min;
    auto corner2 = scene.domain_min;
    auto corner3 = scene.domain_min;
    corner0[fixed_axis] = coordinate;
    corner1[fixed_axis] = coordinate;
    corner2[fixed_axis] = coordinate;
    corner3[fixed_axis] = coordinate;
    corner1[varying_axes[0]] = scene.domain_max[varying_axes[0]];
    corner2[varying_axes[0]] = scene.domain_max[varying_axes[0]];
    corner2[varying_axes[1]] = scene.domain_max[varying_axes[1]];
    corner3[varying_axes[1]] = scene.domain_max[varying_axes[1]];

    lines.push_back({corner0, corner1, color});
    lines.push_back({corner1, corner2, color});
    lines.push_back({corner2, corner3, color});
    lines.push_back({corner3, corner0, color});
}

} // namespace

auto build_grid_lines(const RenderScene &scene) -> std::vector<RenderLine> {
    const auto cells = scene.grid_overlay.cells_per_axis;
    if (!scene.grid_overlay.show_grid || cells <= 0) { return {}; }

    std::vector<RenderLine> lines;
    if (scene.dimension == RuntimeDimension::Two) {
        lines.reserve(static_cast<std::size_t>(2 * (cells + 1)));
        for (int i = 0; i <= cells; ++i) {
            const auto fraction = static_cast<float>(i) / static_cast<float>(cells);
            const auto x = coordinate_at_fraction(scene, 0, fraction);
            const auto y = coordinate_at_fraction(scene, 1, fraction);
            lines.push_back({Eigen::Vector3f{x, scene.domain_min.y(), 0.0F},
                             Eigen::Vector3f{x, scene.domain_max.y(), 0.0F}, kGridColor});
            lines.push_back({Eigen::Vector3f{scene.domain_min.x(), y, 0.0F},
                             Eigen::Vector3f{scene.domain_max.x(), y, 0.0F}, kGridColor});
        }
        return lines;
    }

    lines.reserve(static_cast<std::size_t>(6 * (cells + 1)));
    append_grid_face(lines, scene, 0, 1, 2);
    append_grid_face(lines, scene, 1, 0, 2);
    append_grid_face(lines, scene, 2, 0, 1);
    return lines;
}

auto build_boundary_guide_lines(const RenderScene &scene) -> std::vector<RenderLine> {
    const auto &overlay = scene.grid_overlay;
    if (!overlay.show_boundaries || overlay.cells_per_axis <= 0) { return {}; }

    const int dimensions = scene.dimension == RuntimeDimension::Two ? 2 : 3;
    std::vector<RenderLine> lines;
    lines.reserve(static_cast<std::size_t>((scene.dimension == RuntimeDimension::Two ? 2 : 8) * dimensions));
    for (int axis = 0; axis < dimensions; ++axis) {
        for (int face = 0; face < 2; ++face) {
            const bool lower_face = face == 0;
            const auto mode = lower_face ? overlay.lower[static_cast<std::size_t>(axis)]
                                         : overlay.upper[static_cast<std::size_t>(axis)];
            const auto fraction = guide_fraction(overlay, mode);
            const auto guide_position = lower_face ? fraction : 1.0F - fraction;
            const auto coordinate = coordinate_at_fraction(scene, axis, guide_position);
            const auto color = boundary_color(mode);

            if (scene.dimension == RuntimeDimension::Two) {
                if (axis == 0) {
                    lines.push_back({Eigen::Vector3f{coordinate, scene.domain_min.y(), 0.0F},
                                     Eigen::Vector3f{coordinate, scene.domain_max.y(), 0.0F}, color});
                } else {
                    lines.push_back({Eigen::Vector3f{scene.domain_min.x(), coordinate, 0.0F},
                                     Eigen::Vector3f{scene.domain_max.x(), coordinate, 0.0F}, color});
                }
            } else {
                append_boundary_rectangle(lines, scene, axis, coordinate, color);
            }
        }
    }
    return lines;
}

} // namespace nuclear_mpm::viz
