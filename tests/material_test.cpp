#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/material.hpp>

using Catch::Approx;

TEST_CASE("fixed-corotated dilation creates isotropic stress without shear", "[material]") {
    const nuclear_mpm::Material<2> material = nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, 0.2F);
    nuclear_mpm::Particle<2> particle;
    particle.elastic_deformation = nuclear_mpm::scaled_identity<2>(1.1F);

    const auto stress = nuclear_mpm::kirchhoff_stress(material, particle);

    REQUIRE(stress(0, 1) == Approx(0.0F).margin(1.0e-5F));
    REQUIRE(stress(1, 0) == Approx(0.0F).margin(1.0e-5F));
    REQUIRE(stress(0, 0) == Approx(stress(1, 1)).margin(1.0e-5F));
    REQUIRE(stress(0, 0) > 0.0F);
}

TEST_CASE("weakly compressible liquid has no shear response", "[material]") {
    const nuclear_mpm::Material<2> material = nuclear_mpm::WeaklyCompressibleLiquid<2>(1000.0F);
    nuclear_mpm::Particle<2> particle;
    particle.elastic_deformation << 1.0F, 0.25F,
            0.0F, 1.0F;

    const auto stress = nuclear_mpm::kirchhoff_stress(material, particle);

    REQUIRE(stress.norm() == Approx(0.0F).margin(1.0e-5F));
}

TEST_CASE("liquid deformation projection is isotropic and volume preserving", "[material]") {
    const nuclear_mpm::Material<3> material = nuclear_mpm::WeaklyCompressibleLiquid<3>(1000.0F);
    auto trial = nuclear_mpm::scaled_identity<3>(1.0F);
    trial(0, 0) = 1.2F;
    trial(1, 1) = 0.9F;
    trial(2, 2) = 1.1F;
    nuclear_mpm::Scalar plastic_volume_ratio = 1.0F;

    const auto projected = nuclear_mpm::project_deformation(material, trial, plastic_volume_ratio);

    REQUIRE(projected(0, 0) == Approx(projected(1, 1)).margin(1.0e-6F));
    REQUIRE(projected(1, 1) == Approx(projected(2, 2)).margin(1.0e-6F));
    REQUIRE(projected.determinant() == Approx(trial.determinant()).margin(1.0e-5F));
    const nuclear_mpm::Matrix<3> projected_diagonal = projected.diagonal().asDiagonal();
    REQUIRE((projected - projected_diagonal).norm() < 1.0e-6F);
}

TEST_CASE("snow projection bounds elastic stretch and plastic volume", "[material]") {
    const nuclear_mpm::Material<2> material = nuclear_mpm::Snow<2>(1000.0F, 0.2F);
    nuclear_mpm::Matrix<2> trial = nuclear_mpm::Matrix<2>::Zero();
    trial(0, 0) = 0.8F;
    trial(1, 1) = 1.2F;
    nuclear_mpm::Scalar plastic_volume_ratio = 1.0F;

    const auto projected = nuclear_mpm::project_deformation(material, trial, plastic_volume_ratio);
    const auto decomposition = nuclear_mpm::signed_svd(projected);

    for (int axis = 0; axis < 2; ++axis) {
        REQUIRE(decomposition.singular_values(axis) >= 1.0F - 2.5e-2F);
        REQUIRE(decomposition.singular_values(axis) <= 1.0F + 4.5e-3F);
    }
    REQUIRE(plastic_volume_ratio >= 0.6F);
    REQUIRE(plastic_volume_ratio <= 20.0F);
}

TEST_CASE("material parameters and density are validated", "[material]") {
    REQUIRE_THROWS_AS(nuclear_mpm::FixedCorotatedSolid<2>(0.0F, 0.2F), std::invalid_argument);
    REQUIRE_THROWS_AS(nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, 0.5F), std::invalid_argument);
    REQUIRE_THROWS_AS(nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, -1.0F), std::invalid_argument);
    REQUIRE_THROWS_AS(nuclear_mpm::WeaklyCompressibleLiquid<2>(0.0F), std::invalid_argument);

    const nuclear_mpm::Material<2> material = nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, 0.2F);
    REQUIRE(nuclear_mpm::characteristic_wave_speed(material, 1.0F) > 0.0F);
    REQUIRE_THROWS_AS(nuclear_mpm::characteristic_wave_speed(material, 0.0F), std::invalid_argument);
}

TEST_CASE("material table validates identifiers", "[material]") {
    nuclear_mpm::MaterialTable<2> materials;
    const auto solid_id = materials.add(nuclear_mpm::FixedCorotatedSolid<2>(1000.0F, 0.2F));
    const auto liquid_id = materials.add(nuclear_mpm::WeaklyCompressibleLiquid<2>(500.0F));

    REQUIRE(solid_id == 0);
    REQUIRE(liquid_id == 1);
    REQUIRE(materials.size() == 2);
    REQUIRE_NOTHROW(materials.at(solid_id));
    REQUIRE_THROWS_AS(materials.at(2), std::out_of_range);
}
