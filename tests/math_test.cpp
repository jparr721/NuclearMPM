#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nuclear_mpm/math.hpp>

#include <limits>

using Catch::Approx;

TEST_CASE("scaled identity is complete in 2D and 3D", "[math]") {
    const auto identity_2d = nuclear_mpm::scaled_identity<2>(2.0F);
    const auto identity_3d = nuclear_mpm::scaled_identity<3>(2.0F);

    REQUIRE(identity_2d.determinant() == Approx(4.0F));
    REQUIRE(identity_3d.determinant() == Approx(8.0F));
    REQUIRE(identity_3d(2, 2) == Approx(2.0F));
}

TEST_CASE("filled matrix is distinct from scaled identity", "[math]") {
    const auto filled = nuclear_mpm::filled_matrix<2>(3.0F);

    REQUIRE(filled(0, 0) == Approx(3.0F));
    REQUIRE(filled(0, 1) == Approx(3.0F));
    REQUIRE(filled(1, 0) == Approx(3.0F));
    REQUIRE(filled(1, 1) == Approx(3.0F));
}

TEST_CASE("signed SVD reconstructs reflected 2D input", "[math]") {
    nuclear_mpm::Matrix<2> input;
    input << 1.2F, 0.3F,
            -0.2F, -0.8F;

    const auto result = nuclear_mpm::signed_svd<2>(input);
    const auto reconstructed = result.u * result.singular_values.asDiagonal() * result.v.transpose();

    REQUIRE((reconstructed - input).norm() < 1.0e-5F);
    REQUIRE(result.u.determinant() == Approx(1.0F).margin(1.0e-5F));
    REQUIRE(result.v.determinant() == Approx(1.0F).margin(1.0e-5F));
}

TEST_CASE("signed SVD reconstructs 3D input", "[math]") {
    nuclear_mpm::Matrix<3> input;
    input << 1.0F, 0.2F, 0.0F,
            -0.1F, 0.9F, 0.3F,
            0.0F, -0.2F, 1.1F;

    const auto result = nuclear_mpm::signed_svd<3>(input);
    const auto reconstructed = result.u * result.singular_values.asDiagonal() * result.v.transpose();

    REQUIRE((reconstructed - input).norm() < 1.0e-5F);
    REQUIRE(result.u.determinant() == Approx(1.0F).margin(1.0e-5F));
    REQUIRE(result.v.determinant() == Approx(1.0F).margin(1.0e-5F));
}

TEST_CASE("polar decomposition reconstructs deformation", "[math]") {
    nuclear_mpm::Matrix<2> input;
    input << 1.1F, 0.2F,
            0.1F, 0.9F;

    const auto result = nuclear_mpm::polar_decomposition<2>(input);

    REQUIRE((result.rotation * result.stretch - input).norm() < 1.0e-5F);
    REQUIRE(result.rotation.determinant() == Approx(1.0F).margin(1.0e-5F));
}

TEST_CASE("matrix decompositions reject non-finite input", "[math]") {
    auto input = nuclear_mpm::scaled_identity<2>(1.0F);
    input(0, 1) = std::numeric_limits<nuclear_mpm::Scalar>::infinity();

    REQUIRE_THROWS_AS(nuclear_mpm::signed_svd<2>(input), std::invalid_argument);
    REQUIRE_THROWS_AS(nuclear_mpm::polar_decomposition<2>(input), std::invalid_argument);
}
