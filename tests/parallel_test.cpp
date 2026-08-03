#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/parallel.hpp>
#include <nuclear_mpm/sampling.hpp>
#include <nuclear_mpm/simulation.hpp>

TEST_CASE("parallel worker selection obeys scratch memory budget", "[parallel]") {
    constexpr std::size_t mib = 1024U * 1024U;

    REQUIRE(nuclear_mpm::select_parallel_worker_count(16U * mib, 8, 64U * mib) == 4);
    REQUIRE(nuclear_mpm::select_parallel_worker_count(40U * mib, 8, 64U * mib) == 1);
    REQUIRE(nuclear_mpm::select_parallel_worker_count(80U * mib, 8, 64U * mib) == 1);
    REQUIRE(nuclear_mpm::select_parallel_worker_count(16U * mib, 1, 64U * mib) == 1);
}

TEST_CASE("parallel configuration validates worker and budget limits", "[parallel]") {
    nuclear_mpm::ParallelConfig config;
    REQUIRE_NOTHROW(nuclear_mpm::validate_parallel_config(config));

    config.max_workers = 0;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_parallel_config(config), std::invalid_argument);

    config.max_workers = 2;
    config.scratch_memory_budget_bytes = 0;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_parallel_config(config), std::invalid_argument);
}

TEST_CASE("parallel-enabled transfer agrees with serial reference", "[parallel]") {
    nuclear_mpm::SimulationConfig<2> serial_config;
    serial_config.cells_per_axis = 8;
    serial_config.materials.add(nuclear_mpm::FixedCorotatedSolid<2>(100.0F, 0.2F));
    auto parallel_config = serial_config;
    parallel_config.parallel.enabled = true;
    parallel_config.parallel.max_workers = 2;
    parallel_config.parallel.scratch_memory_budget_bytes = 1024U * 1024U;

    auto particles = nuclear_mpm::sample_box_lattice<2>(
            nuclear_mpm::Index<2>{2, 2}, nuclear_mpm::Vector<2>{0.4F, 0.4F},
            nuclear_mpm::Vector<2>{0.6F, 0.6F}, 1.0F, 0, 0);
    particles[0].velocity << 0.2F, -0.1F;
    particles[1].velocity << -0.1F, 0.3F;

    nuclear_mpm::Simulation<2> serial(serial_config, {}, particles);
    nuclear_mpm::Simulation<2> parallel(parallel_config, {}, particles);
    serial.substep(1.0e-5F);
    parallel.substep(1.0e-5F);

    REQUIRE(serial.particles().size() == parallel.particles().size());
    for (std::size_t index = 0; index < serial.particles().size(); ++index) {
        REQUIRE((serial.particles()[index].position - parallel.particles()[index].position).norm() < 1.0e-6F);
        REQUIRE((serial.particles()[index].velocity - parallel.particles()[index].velocity).norm() < 1.0e-5F);
        REQUIRE((serial.particles()[index].elastic_deformation -
                 parallel.particles()[index].elastic_deformation)
                        .norm() < 1.0e-5F);
    }
}
