#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/stencil.hpp>

using Catch::Approx;

TEMPLATE_TEST_CASE_SIG("quadratic stencil reproduces constants and first moments", "[stencil]", ((int Dim), Dim), 2, 3) {
    constexpr int cells_per_axis = 8;
    constexpr nuclear_mpm::Scalar inverse_cell_size = static_cast<nuclear_mpm::Scalar>(cells_per_axis);
    const auto position = nuclear_mpm::Vector<Dim>::Constant(0.37F);
    const nuclear_mpm::QuadraticStencil<Dim> stencil(position, inverse_cell_size, cells_per_axis + 1);

    nuclear_mpm::Scalar weight_sum = 0.0F;
    nuclear_mpm::Vector<Dim> weighted_grid_displacement = nuclear_mpm::Vector<Dim>::Zero();
    std::size_t visited = 0;

    stencil.for_each_node([&](const nuclear_mpm::StencilNode<Dim> &node) {
        ++visited;
        weight_sum += node.weight;
        weighted_grid_displacement += node.weight * node.grid_displacement;
        for (int axis = 0; axis < Dim; ++axis) {
            REQUIRE(node.index(axis) >= 0);
            REQUIRE(node.index(axis) < cells_per_axis + 1);
        }
    });

    std::size_t expected_nodes = 1;
    for (int axis = 0; axis < Dim; ++axis) { expected_nodes *= 3; }
    REQUIRE(visited == expected_nodes);
    REQUIRE(weight_sum == Approx(1.0F).margin(1.0e-6F));
    REQUIRE(weighted_grid_displacement.norm() < 1.0e-6F);
}

TEST_CASE("quadratic stencil includes the final valid grid node", "[stencil]") {
    constexpr int cells_per_axis = 8;
    constexpr nuclear_mpm::Scalar cell_size = 1.0F / cells_per_axis;
    const nuclear_mpm::Vector<2> position{1.0F - cell_size, 1.0F - cell_size};
    const nuclear_mpm::QuadraticStencil<2> stencil(position, 1.0F / cell_size, cells_per_axis + 1);

    REQUIRE(stencil.base() == nuclear_mpm::Index<2>{6, 6});

    bool visited_last_node = false;
    stencil.for_each_node([&](const nuclear_mpm::StencilNode<2> &node) {
        if (node.index == nuclear_mpm::Index<2>{cells_per_axis, cells_per_axis}) {
            visited_last_node = true;
        }
    });

    REQUIRE(visited_last_node);
}

TEST_CASE("quadratic stencil uses floor and rejects incomplete support", "[stencil]") {
    constexpr int cells_per_axis = 8;
    const auto inverse_cell_size = static_cast<nuclear_mpm::Scalar>(cells_per_axis);

    REQUIRE_THROWS_AS(
            nuclear_mpm::QuadraticStencil<2>(nuclear_mpm::Vector<2>{0.0F, 0.5F}, inverse_cell_size,
                                             cells_per_axis + 1),
            std::out_of_range);
    REQUIRE_THROWS_AS(
            nuclear_mpm::QuadraticStencil<2>(nuclear_mpm::Vector<2>{1.0F, 0.5F}, inverse_cell_size,
                                             cells_per_axis + 1),
            std::out_of_range);
}

TEST_CASE("quadratic stencil validates construction parameters", "[stencil]") {
    REQUIRE_THROWS_AS(
            nuclear_mpm::QuadraticStencil<2>(nuclear_mpm::Vector<2>{0.5F, 0.5F}, 0.0F, 9),
            std::invalid_argument);
    REQUIRE_THROWS_AS(
            nuclear_mpm::QuadraticStencil<2>(nuclear_mpm::Vector<2>{0.5F, 0.5F}, 8.0F, 2),
            std::invalid_argument);
}
