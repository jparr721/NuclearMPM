#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/simulation.hpp>

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
