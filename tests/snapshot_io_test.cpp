#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "snapshot_io.h"

#include <filesystem>

using Catch::Approx;

TEST_CASE("snapshot round-trips 2D particle and grid state", "[io]") {
    nuclear_mpm::SimulationConfig<2> config;
    config.cells_per_axis = 2;
    config.materials.add(nuclear_mpm::FixedCorotatedSolid<2>(100.0F, 0.2F));
    auto particles = nuclear_mpm::sample_box_lattice<2>(
            nuclear_mpm::Index<2>{2, 2}, nuclear_mpm::Vector<2>{0.2F, 0.2F},
            nuclear_mpm::Vector<2>{0.6F, 0.6F}, 1.0F, 0, 42U);
    nuclear_mpm::Grid<2> grid(config.cells_per_axis);
    auto &last_node = grid.node(nuclear_mpm::Index<2>{2, 2});
    last_node.mass = 3.0F;
    last_node.momentum << 1.0F, 2.0F;
    last_node.velocity << 4.0F, 5.0F;
    nuclear_mpm::StepDiagnostics<2> diagnostics;
    diagnostics.substep_count = 2;
    diagnostics.particle_mass = 0.16F;

    const auto path = std::filesystem::temp_directory_path() / "nuclear_mpm_snapshot_2d.txt";
    nuclear_mpm::io::write_snapshot(path, 0.25F, config, particles, grid, diagnostics);
    const auto loaded = nuclear_mpm::io::read_snapshot<2>(path);
    std::filesystem::remove(path);

    REQUIRE(loaded.version == 1);
    REQUIRE(loaded.simulation_time == Approx(0.25F));
    REQUIRE(loaded.cells_per_axis == 2);
    REQUIRE(loaded.particles.size() == particles.size());
    REQUIRE(loaded.grid_nodes.size() == 9);
    REQUIRE(loaded.particles.back().user_tag == 42U);
    REQUIRE((loaded.particles.front().position - particles.front().position).norm() < 1.0e-6F);

    const auto &loaded_last = loaded.grid_nodes.back();
    REQUIRE(loaded_last.coordinate == nuclear_mpm::Index<2>{2, 2});
    REQUIRE(loaded_last.mass == Approx(3.0F));
    REQUIRE(loaded_last.momentum(0) == Approx(1.0F));
    REQUIRE(loaded_last.momentum(1) == Approx(2.0F));
    REQUIRE(loaded_last.velocity(0) == Approx(4.0F));
    REQUIRE(loaded_last.velocity(1) == Approx(5.0F));
    REQUIRE(loaded.diagnostics.substep_count == 2);
}

TEST_CASE("snapshot reader rejects the wrong dimension", "[io]") {
    nuclear_mpm::SimulationConfig<2> config;
    config.cells_per_axis = 2;
    config.materials.add(nuclear_mpm::FixedCorotatedSolid<2>(100.0F, 0.2F));
    nuclear_mpm::Grid<2> grid(config.cells_per_axis);
    const std::vector<nuclear_mpm::Particle<2>> particles;
    const nuclear_mpm::StepDiagnostics<2> diagnostics;
    const auto path = std::filesystem::temp_directory_path() / "nuclear_mpm_snapshot_dimension.txt";
    nuclear_mpm::io::write_snapshot(path, 0.0F, config, particles, grid, diagnostics);

    REQUIRE_THROWS_AS(nuclear_mpm::io::read_snapshot<3>(path), std::runtime_error);
    std::filesystem::remove(path);
}
