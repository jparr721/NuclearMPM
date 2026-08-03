#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "camera.h"

using Catch::Approx;

namespace {

auto unit_bounds() -> nuclear_mpm::viz::DomainBounds {
    return {Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones()};
}

} // namespace

TEST_CASE("orthographic camera reset frames finite unit domain", "[camera]") {
    nuclear_mpm::viz::OrthographicCamera camera;
    camera.reset(unit_bounds());

    REQUIRE((camera.target() - Eigen::Vector3f::Constant(0.5F)).norm() < 1.0e-6F);
    REQUIRE(camera.view_matrix().allFinite());
    REQUIRE(camera.projection_matrix(1.0F).allFinite());
    REQUIRE(camera.half_height() > 0.5F);
}

TEST_CASE("orbit camera rotation preserves target distance", "[camera]") {
    nuclear_mpm::viz::OrbitCamera camera;
    camera.reset(unit_bounds());
    const auto distance = camera.distance();

    camera.orbit(Eigen::Vector2f{0.4F, -0.2F});

    REQUIRE(camera.distance() == Approx(distance));
    REQUIRE((camera.eye() - camera.target()).norm() == Approx(distance).margin(1.0e-5F));
    REQUIRE(camera.view_matrix().allFinite());
}

TEST_CASE("orbit camera pan translates eye and target equally", "[camera]") {
    nuclear_mpm::viz::OrbitCamera camera;
    camera.reset(unit_bounds());
    const auto old_eye = camera.eye();
    const auto old_target = camera.target();

    camera.pan(Eigen::Vector2f{0.1F, -0.2F});

    REQUIRE(((camera.eye() - old_eye) - (camera.target() - old_target)).norm() < 1.0e-6F);
}

TEST_CASE("orbit camera dolly keeps a positive bounded distance", "[camera]") {
    nuclear_mpm::viz::OrbitCamera camera;
    camera.reset(unit_bounds());
    const auto original = camera.distance();

    camera.dolly(1.0F);
    REQUIRE(camera.distance() < original);
    camera.dolly(1000.0F);
    REQUIRE(camera.distance() >= 0.05F);
    camera.dolly(-1000.0F);
    REQUIRE(camera.distance() <= 100.0F);
}

TEST_CASE("camera projection responds to aspect ratio", "[camera]") {
    nuclear_mpm::viz::OrbitCamera camera;
    camera.reset(unit_bounds());

    const auto square = camera.projection_matrix(1.0F);
    const auto wide = camera.projection_matrix(2.0F);

    REQUIRE(square.allFinite());
    REQUIRE(wide.allFinite());
    REQUIRE(square(0, 0) != Approx(wide(0, 0)));
}

TEST_CASE("UI mouse capture neutralizes camera input", "[camera]") {
    const nuclear_mpm::viz::CameraInput input{0.1F, 0.2F, 0.3F, 0.4F, 1.0F};

    const auto captured = nuclear_mpm::viz::filter_camera_input(input, true);
    const auto active = nuclear_mpm::viz::filter_camera_input(input, false);

    REQUIRE(captured == nuclear_mpm::viz::CameraInput{});
    REQUIRE(active == input);
}
