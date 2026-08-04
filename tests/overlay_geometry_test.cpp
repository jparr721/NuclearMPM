#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "overlay_geometry.h"

#include <array>
#include <utility>

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

TEST_CASE("3D overlays map lower faces and boundary guides onto a custom domain", "[overlay]") {
    nuclear_mpm::viz::RenderScene scene;
    scene.dimension = nuclear_mpm::RuntimeDimension::Three;
    scene.domain_min = Eigen::Vector3f{-2.0F, 10.0F, 4.0F};
    scene.domain_max = Eigen::Vector3f{2.0F, 18.0F, 12.0F};
    scene.grid_overlay.cells_per_axis = 4;
    scene.grid_overlay.boundary_layers = 1;
    scene.grid_overlay.lower[0] = nuclear_mpm::BoundaryMode::Sticky;

    const auto grid_lines = nuclear_mpm::viz::build_grid_lines(scene);
    REQUIRE(grid_lines.at(2).start.isApprox(Eigen::Vector3f{-2.0F, 12.0F, 4.0F}));
    REQUIRE(grid_lines.at(2).end.isApprox(Eigen::Vector3f{-2.0F, 12.0F, 12.0F}));
    REQUIRE(grid_lines.at(14).start.isApprox(Eigen::Vector3f{0.0F, 10.0F, 4.0F}));
    REQUIRE(grid_lines.at(14).end.isApprox(Eigen::Vector3f{0.0F, 10.0F, 12.0F}));
    REQUIRE(grid_lines.at(26).start.isApprox(Eigen::Vector3f{1.0F, 10.0F, 4.0F}));
    REQUIRE(grid_lines.at(26).end.isApprox(Eigen::Vector3f{1.0F, 18.0F, 4.0F}));

    const auto guide_lines = nuclear_mpm::viz::build_boundary_guide_lines(scene);
    const Eigen::Vector3f corner0{-0.5F, 10.0F, 4.0F};
    const Eigen::Vector3f corner1{-0.5F, 18.0F, 4.0F};
    const Eigen::Vector3f corner2{-0.5F, 18.0F, 12.0F};
    const Eigen::Vector3f corner3{-0.5F, 10.0F, 12.0F};
    const std::array expected_edges{
            std::pair{corner0, corner1}, std::pair{corner1, corner2},
            std::pair{corner2, corner3}, std::pair{corner3, corner0}};
    for (std::size_t i = 0; i < expected_edges.size(); ++i) {
        REQUIRE(guide_lines.at(i).start.isApprox(expected_edges[i].first));
        REQUIRE(guide_lines.at(i).end.isApprox(expected_edges[i].second));
        REQUIRE(guide_lines.at(i).color == nuclear_mpm::viz::kStickyBoundaryColor);
    }
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
