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
