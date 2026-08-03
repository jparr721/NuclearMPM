#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/boundary.hpp>
#include <nuclear_mpm/config.hpp>
#include <nuclear_mpm/diagnostics.hpp>

#include <limits>

TEST_CASE("slip boundary removes only outward normal velocity", "[boundary]") {
    nuclear_mpm::Grid<2> grid(8);
    auto &node = grid.node(nuclear_mpm::Index<2>{1, 4});
    node.mass = 2.0F;
    node.velocity << -3.0F, 4.0F;

    nuclear_mpm::DomainBoundaries<2> boundaries;
    boundaries.boundary_layers = 2;
    boundaries.lower[0] = nuclear_mpm::BoundaryMode::Slip;

    nuclear_mpm::apply_grid_boundaries(grid, boundaries);

    REQUIRE(node.mass == 2.0F);
    REQUIRE(node.velocity(0) == 0.0F);
    REQUIRE(node.velocity(1) == 4.0F);
}

TEST_CASE("sticky boundary removes all contact velocity without erasing mass", "[boundary]") {
    nuclear_mpm::Grid<2> grid(8);
    auto &node = grid.node(nuclear_mpm::Index<2>{7, 4});
    node.mass = 2.0F;
    node.velocity << 3.0F, 4.0F;

    nuclear_mpm::DomainBoundaries<2> boundaries;
    boundaries.boundary_layers = 2;
    boundaries.upper[0] = nuclear_mpm::BoundaryMode::Sticky;

    nuclear_mpm::apply_grid_boundaries(grid, boundaries);

    REQUIRE(node.mass == 2.0F);
    REQUIRE(node.velocity.isZero());
}

TEST_CASE("outflow boundary leaves outward grid velocity unchanged", "[boundary]") {
    nuclear_mpm::Grid<2> grid(8);
    auto &node = grid.node(nuclear_mpm::Index<2>{1, 4});
    node.mass = 2.0F;
    node.velocity << -3.0F, 4.0F;

    nuclear_mpm::DomainBoundaries<2> boundaries;
    boundaries.boundary_layers = 2;
    boundaries.lower[0] = nuclear_mpm::BoundaryMode::Outflow;

    nuclear_mpm::apply_grid_boundaries(grid, boundaries);

    REQUIRE(node.mass == 2.0F);
    REQUIRE(node.velocity(0) == -3.0F);
    REQUIRE(node.velocity(1) == 4.0F);
}

TEST_CASE("simulation configuration validates numerical invariants", "[config]") {
    nuclear_mpm::SimulationConfig<2> config;
    config.materials.add(nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, 0.2F));
    REQUIRE_NOTHROW(nuclear_mpm::validate_simulation_config(config));

    auto invalid = config;
    invalid.cells_per_axis = 0;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(invalid), std::invalid_argument);

    invalid = config;
    invalid.max_time_step = 0.0F;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(invalid), std::invalid_argument);

    invalid = config;
    invalid.cfl_factor = 1.1F;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(invalid), std::invalid_argument);

    invalid = config;
    invalid.boundaries.boundary_layers = 0;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(invalid), std::invalid_argument);

    invalid = config;
    invalid.max_substeps_per_advance = 0;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(invalid), std::invalid_argument);
}

TEST_CASE("simulation configuration requires a material", "[config]") {
    const nuclear_mpm::SimulationConfig<3> config;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_simulation_config(config), std::invalid_argument);
}

TEST_CASE("runtime controls require finite gravity", "[config]") {
    nuclear_mpm::RuntimeControls<2> controls;
    REQUIRE_NOTHROW(nuclear_mpm::validate_runtime_controls(controls));

    controls.gravity(1) = std::numeric_limits<nuclear_mpm::Scalar>::quiet_NaN();
    REQUIRE_THROWS_AS(nuclear_mpm::validate_runtime_controls(controls), std::invalid_argument);
}

TEST_CASE("simulation errors preserve phase and entity context", "[config]") {
    const nuclear_mpm::SimulationError error(
            nuclear_mpm::SimulationPhase::ParticleToGrid, "invalid stencil", 7);

    REQUIRE(error.phase() == nuclear_mpm::SimulationPhase::ParticleToGrid);
    REQUIRE(error.entity_index().has_value());
    REQUIRE(*error.entity_index() == 7);
    REQUIRE(std::string(error.what()).find("invalid stencil") != std::string::npos);
}
