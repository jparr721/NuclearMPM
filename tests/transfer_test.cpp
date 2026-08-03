#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/simulation.hpp>

using Catch::Approx;

namespace {

template<int Dim>
auto solid_config() -> nuclear_mpm::SimulationConfig<Dim> {
    nuclear_mpm::SimulationConfig<Dim> config;
    config.cells_per_axis = 8;
    config.max_time_step = 1.0e-4F;
    config.materials.add(nuclear_mpm::FixedCorotatedSolid<Dim>(100.0F, 0.2F));
    return config;
}

template<int Dim>
auto particle_at(const nuclear_mpm::Vector<Dim> &position) -> nuclear_mpm::Particle<Dim> {
    nuclear_mpm::Particle<Dim> particle;
    particle.position = position;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    return particle;
}

} // namespace

TEST_CASE("P2G conserves particle mass", "[transfer]") {
    auto first = particle_at<2>(nuclear_mpm::Vector<2>{0.45F, 0.5F});
    auto second = particle_at<2>(nuclear_mpm::Vector<2>{0.55F, 0.5F});
    second.mass = 2.0F;
    second.reference_volume = 2.0F;

    nuclear_mpm::Simulation<2> simulation(solid_config<2>(), {}, {first, second});
    const auto diagnostics = simulation.substep(1.0e-5F);

    nuclear_mpm::Scalar grid_mass = 0.0F;
    simulation.grid().for_each_node(
            [&](const nuclear_mpm::Index<2> &, const nuclear_mpm::GridNode<2> &node) { grid_mass += node.mass; });

    REQUIRE(grid_mass == Approx(3.0F).margin(1.0e-5F));
    REQUIRE(diagnostics.grid_mass == Approx(3.0F).margin(1.0e-5F));
}

TEST_CASE("force-free transfer conserves linear momentum", "[transfer]") {
    auto first = particle_at<2>(nuclear_mpm::Vector<2>{0.45F, 0.5F});
    auto second = particle_at<2>(nuclear_mpm::Vector<2>{0.55F, 0.5F});
    first.velocity << 1.0F, -0.5F;
    second.velocity << -0.25F, 0.75F;
    const nuclear_mpm::Vector<2> initial_momentum = first.mass * first.velocity + second.mass * second.velocity;

    nuclear_mpm::Simulation<2> simulation(solid_config<2>(), {}, {first, second});
    simulation.substep(1.0e-5F);

    nuclear_mpm::Vector<2> final_momentum = nuclear_mpm::Vector<2>::Zero();
    for (const auto &particle : simulation.particles()) { final_momentum += particle.mass * particle.velocity; }

    REQUIRE((final_momentum - initial_momentum).norm() < 1.0e-5F);
}

TEST_CASE("resting particle remains at rest without forces", "[transfer]") {
    const auto initial = particle_at<2>(nuclear_mpm::Vector<2>{0.47F, 0.53F});
    nuclear_mpm::Simulation<2> simulation(solid_config<2>(), {}, {initial});

    simulation.substep(1.0e-5F);
    const auto &particle = simulation.particles().front();

    REQUIRE(particle.position(0) == Approx(initial.position(0)).margin(1.0e-7F));
    REQUIRE(particle.position(1) == Approx(initial.position(1)).margin(1.0e-7F));
    REQUIRE(particle.velocity.norm() < 1.0e-7F);
    REQUIRE((particle.elastic_deformation - nuclear_mpm::scaled_identity<2>(1.0F)).norm() < 1.0e-7F);
}

TEST_CASE("G2P reproduces uniform velocity", "[transfer]") {
    auto initial = particle_at<2>(nuclear_mpm::Vector<2>{0.47F, 0.53F});
    initial.velocity << 1.25F, -0.75F;
    nuclear_mpm::Simulation<2> simulation(solid_config<2>(), {}, {initial});

    simulation.substep(1.0e-5F);

    REQUIRE((simulation.particles().front().velocity - initial.velocity).norm() < 1.0e-5F);
}

TEST_CASE("APIC transfer reproduces an affine velocity field", "[transfer]") {
    auto initial = particle_at<2>(nuclear_mpm::Vector<2>{0.47F, 0.53F});
    initial.velocity << 0.3F, -0.2F;
    initial.affine_velocity << 0.2F, -0.1F,
            0.15F, 0.05F;
    nuclear_mpm::Simulation<2> simulation(solid_config<2>(), {}, {initial});

    simulation.substep(1.0e-5F);
    const auto &particle = simulation.particles().front();

    REQUIRE((particle.velocity - initial.velocity).norm() < 1.0e-5F);
    REQUIRE((particle.affine_velocity - initial.affine_velocity).norm() < 1.0e-4F);
}
