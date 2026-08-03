#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/grid.hpp>

#include <set>

TEST_CASE("grid coordinates and flat indices are bijective in 2D", "[grid]") {
    nuclear_mpm::Grid<2> grid(2);
    std::set<std::size_t> visited;

    REQUIRE(grid.cells_per_axis() == 2);
    REQUIRE(grid.nodes_per_axis() == 3);
    REQUIRE(grid.node_count() == 9);

    for (int x = 0; x < grid.nodes_per_axis(); ++x) {
        for (int y = 0; y < grid.nodes_per_axis(); ++y) {
            const nuclear_mpm::Index<2> coordinate{x, y};
            const auto flat = grid.flatten(coordinate);
            visited.insert(flat);
            REQUIRE(grid.coordinate(flat) == coordinate);
        }
    }

    REQUIRE(visited.size() == grid.node_count());
}

TEST_CASE("grid coordinates and flat indices are bijective in 3D", "[grid]") {
    nuclear_mpm::Grid<3> grid(2);
    std::set<std::size_t> visited;

    REQUIRE(grid.nodes_per_axis() == 3);
    REQUIRE(grid.node_count() == 27);

    for (int x = 0; x < grid.nodes_per_axis(); ++x) {
        for (int y = 0; y < grid.nodes_per_axis(); ++y) {
            for (int z = 0; z < grid.nodes_per_axis(); ++z) {
                const nuclear_mpm::Index<3> coordinate{x, y, z};
                const auto flat = grid.flatten(coordinate);
                visited.insert(flat);
                REQUIRE(grid.coordinate(flat) == coordinate);
            }
        }
    }

    REQUIRE(visited.size() == grid.node_count());
}

TEST_CASE("grid validates coordinates and resolution", "[grid]") {
    REQUIRE_THROWS_AS(nuclear_mpm::Grid<2>(0), std::invalid_argument);

    nuclear_mpm::Grid<2> grid(2);
    REQUIRE_THROWS_AS(grid.flatten(nuclear_mpm::Index<2>{-1, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(grid.flatten(nuclear_mpm::Index<2>{3, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(grid.coordinate(grid.node_count()), std::out_of_range);
}

TEST_CASE("grid reset clears mass momentum and velocity", "[grid]") {
    nuclear_mpm::Grid<2> grid(2);
    auto &node = grid.node(nuclear_mpm::Index<2>{1, 1});
    node.mass = 3.0F;
    node.momentum << 1.0F, 2.0F;
    node.velocity << 4.0F, 5.0F;

    grid.reset();

    REQUIRE(node.mass == 0.0F);
    REQUIRE(node.momentum.isZero());
    REQUIRE(node.velocity.isZero());
}
