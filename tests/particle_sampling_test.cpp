#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/particle.hpp>
#include <nuclear_mpm/sampling.hpp>

#include <limits>

using Catch::Approx;

TEST_CASE("2D box sampling derives physical mass and volume", "[sampling]") {
    const auto particles = nuclear_mpm::sample_box_lattice<2>(
            nuclear_mpm::Index<2>{2, 3}, nuclear_mpm::Vector<2>{0.2F, 0.3F},
            nuclear_mpm::Vector<2>{0.6F, 0.9F}, 2.0F, 4, 0xAABBCCU);

    REQUIRE(particles.size() == 6);

    nuclear_mpm::Scalar total_volume = 0.0F;
    nuclear_mpm::Scalar total_mass = 0.0F;
    for (const auto &particle : particles) {
        REQUIRE(particle.position(0) > 0.2F);
        REQUIRE(particle.position(0) < 0.6F);
        REQUIRE(particle.position(1) > 0.3F);
        REQUIRE(particle.position(1) < 0.9F);
        REQUIRE(particle.material_id == 4);
        REQUIRE(particle.user_tag == 0xAABBCCU);
        REQUIRE(particle.elastic_deformation.isIdentity());
        total_volume += particle.reference_volume;
        total_mass += particle.mass;
    }

    REQUIRE(total_volume == Approx(0.24F).margin(1.0e-6F));
    REQUIRE(total_mass == Approx(0.48F).margin(1.0e-6F));
}

TEST_CASE("3D box sampling derives physical mass and volume", "[sampling]") {
    const auto particles = nuclear_mpm::sample_box_lattice<3>(
            nuclear_mpm::Index<3>{2, 2, 2}, nuclear_mpm::Vector<3>{0.2F, 0.2F, 0.2F},
            nuclear_mpm::Vector<3>{0.6F, 0.6F, 0.6F}, 1.5F, 1, 7U);

    REQUIRE(particles.size() == 8);

    nuclear_mpm::Scalar total_volume = 0.0F;
    nuclear_mpm::Scalar total_mass = 0.0F;
    for (const auto &particle : particles) {
        nuclear_mpm::validate_particle(particle, 0);
        REQUIRE(particle.elastic_deformation.determinant() == Approx(1.0F));
        total_volume += particle.reference_volume;
        total_mass += particle.mass;
    }

    REQUIRE(total_volume == Approx(0.064F).margin(1.0e-6F));
    REQUIRE(total_mass == Approx(0.096F).margin(1.0e-6F));
}

TEST_CASE("box sampling rejects invalid geometry and density", "[sampling]") {
    REQUIRE_THROWS_AS(
            nuclear_mpm::sample_box_lattice<2>(nuclear_mpm::Index<2>{0, 2},
                                                nuclear_mpm::Vector<2>{0.2F, 0.2F},
                                                nuclear_mpm::Vector<2>{0.6F, 0.6F}, 1.0F, 0, 0),
            std::invalid_argument);
    REQUIRE_THROWS_AS(
            nuclear_mpm::sample_box_lattice<2>(nuclear_mpm::Index<2>{2, 2},
                                                nuclear_mpm::Vector<2>{0.6F, 0.2F},
                                                nuclear_mpm::Vector<2>{0.2F, 0.6F}, 1.0F, 0, 0),
            std::invalid_argument);
    REQUIRE_THROWS_AS(
            nuclear_mpm::sample_box_lattice<2>(nuclear_mpm::Index<2>{2, 2},
                                                nuclear_mpm::Vector<2>{0.2F, 0.2F},
                                                nuclear_mpm::Vector<2>{0.6F, 0.6F}, 0.0F, 0, 0),
            std::invalid_argument);
}

TEST_CASE("particle validation rejects invalid physical state", "[sampling]") {
    nuclear_mpm::Particle<2> particle;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;

    REQUIRE_NOTHROW(nuclear_mpm::validate_particle(particle, 3));

    particle.mass = 0.0F;
    REQUIRE_THROWS_AS(nuclear_mpm::validate_particle(particle, 3), std::invalid_argument);

    particle.mass = 1.0F;
    particle.position(0) = std::numeric_limits<nuclear_mpm::Scalar>::quiet_NaN();
    REQUIRE_THROWS_AS(nuclear_mpm::validate_particle(particle, 3), std::invalid_argument);
}
