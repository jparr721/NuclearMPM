#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/simulation.hpp>

using Catch::Approx;

namespace {

template<int Dim>
auto make_config() -> nuclear_mpm::SimulationConfig<Dim> {
    nuclear_mpm::SimulationConfig<Dim> config;
    config.cells_per_axis = 8;
    config.max_time_step = 1.0e-4F;
    config.materials.add(nuclear_mpm::FixedCorotatedSolid<Dim>(100.0F, 0.2F));
    return config;
}

} // namespace

TEST_CASE("non-empty 3D simulation completes one finite substep", "[simulation]") {
    nuclear_mpm::Particle<3> particle;
    particle.position << 0.5F, 0.5F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;

    nuclear_mpm::Simulation<3> simulation(make_config<3>(), {}, {particle});
    const auto diagnostics = simulation.substep(1.0e-5F);

    REQUIRE(simulation.particles().size() == 1);
    REQUIRE(simulation.particles().front().position.allFinite());
    REQUIRE(simulation.particles().front().elastic_deformation.allFinite());
    REQUIRE(diagnostics.substep_count == 1);
}

TEST_CASE("simulation rejects an initial particle with incomplete grid support", "[simulation]") {
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.0F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;

    REQUIRE_THROWS_AS(nuclear_mpm::Simulation<2>(make_config<2>(), {}, {particle}),
                      nuclear_mpm::SimulationError);
}

TEST_CASE("simulation validates material identifiers", "[simulation]") {
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.5F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    particle.material_id = 3;

    REQUIRE_THROWS_AS(nuclear_mpm::Simulation<2>(make_config<2>(), {}, {particle}), std::invalid_argument);
}

TEST_CASE("simulation validates and updates runtime controls", "[simulation]") {
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.5F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    nuclear_mpm::Simulation<2> simulation(make_config<2>(), {}, {particle});

    nuclear_mpm::RuntimeControls<2> controls;
    controls.gravity << 0.0F, -9.81F;
    simulation.set_runtime_controls(controls);

    REQUIRE(simulation.runtime_controls().gravity == controls.gravity);
}

TEST_CASE("stable timestep obeys transport and material CFL limit", "[simulation]") {
    auto config = make_config<2>();
    config.max_time_step = 1.0e-3F;
    config.cfl_factor = 0.4F;
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.5F, 0.5F;
    particle.velocity << 1000.0F, 0.0F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    nuclear_mpm::Simulation<2> simulation(config, {}, {particle});

    const auto time_step = simulation.stable_time_step();
    const auto density = particle.mass / particle.reference_volume;
    const auto wave_speed = nuclear_mpm::characteristic_wave_speed(config.materials.at(0), density);

    REQUIRE(time_step <= config.max_time_step);
    REQUIRE(time_step <= config.cfl_factor * (1.0F / config.cells_per_axis) /
                                 (particle.velocity.norm() + wave_speed));
}

TEST_CASE("advance performs bounded free fall and reports elapsed time", "[simulation]") {
    auto config = make_config<2>();
    config.max_time_step = 1.0e-4F;
    nuclear_mpm::RuntimeControls<2> controls;
    controls.gravity << 0.0F, -10.0F;
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.5F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    nuclear_mpm::Simulation<2> simulation(config, controls, {particle});

    const auto diagnostics = simulation.advance(4.0e-4F);

    REQUIRE(diagnostics.substep_count == 4);
    REQUIRE(simulation.simulation_time() == Approx(4.0e-4F).margin(1.0e-7F));
    REQUIRE(simulation.particles().front().velocity(1) == Approx(-4.0e-3F).margin(1.0e-6F));
    REQUIRE(simulation.particles().front().position.allFinite());
    REQUIRE(simulation.particles().front().elastic_deformation.allFinite());
}

TEST_CASE("outflow removes particles before stencil support is lost", "[simulation]") {
    auto config = make_config<2>();
    config.boundaries.lower[0] = nuclear_mpm::BoundaryMode::Outflow;
    config.boundaries.outflow_deletion_layers = 2;
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.24F, 0.5F;
    particle.mass = 2.0F;
    particle.reference_volume = 2.0F;
    nuclear_mpm::Simulation<2> simulation(config, {}, {particle});

    const auto diagnostics = simulation.advance(1.0e-4F);

    REQUIRE(simulation.particles().empty());
    REQUIRE(diagnostics.removed_outflow_particles == 1);
    REQUIRE(diagnostics.removed_outflow_mass == Approx(2.0F));
}

TEST_CASE("advance enforces its configured substep ceiling", "[simulation]") {
    auto config = make_config<2>();
    config.max_time_step = 1.0e-4F;
    config.max_substeps_per_advance = 2;
    nuclear_mpm::Particle<2> particle;
    particle.position << 0.5F, 0.5F;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    nuclear_mpm::Simulation<2> simulation(config, {}, {particle});

    REQUIRE_THROWS_AS(simulation.advance(3.0e-4F), nuclear_mpm::SimulationError);
}
