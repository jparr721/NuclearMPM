#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "camera.h"

using Catch::Approx;

namespace {

auto unit_bounds() -> nuclear_mpm::viz::DomainBounds {
    return {Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones()};
}

auto project_to_ndc(const nuclear_mpm::viz::OrbitCamera &camera, const Eigen::Vector3f &point)
        -> Eigen::Vector2f {
    const Eigen::Vector4f homogeneous{point.x(), point.y(), point.z(), 1.0F};
    const Eigen::Vector4f clip = camera.projection_matrix(1.0F) * camera.view_matrix() * homogeneous;
    return clip.head<2>() / clip.w();
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

TEST_CASE("pointer orbit drag makes the scene follow the cursor", "[camera]") {
    const auto input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 10.0F}, true, false, 0.0F);

    REQUIRE(input.orbit_x == Approx(0.1F));
    REQUIRE(input.orbit_y == Approx(0.05F));
    REQUIRE(input.pan_x == 0.0F);
    REQUIRE(input.pan_y == 0.0F);
}

TEST_CASE("pointer pan and scroll preserve grab-style mapping", "[camera]") {
    const auto input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 10.0F}, false, true, 2.0F);

    REQUIRE(input.pan_x == Approx(-0.02F));
    REQUIRE(input.pan_y == Approx(0.01F));
    REQUIRE(input.dolly == Approx(0.3F));
    REQUIRE(input.orbit_x == 0.0F);
    REQUIRE(input.orbit_y == 0.0F);
}

TEST_CASE("mapped orbit drag moves projected scene features with the cursor", "[camera]") {
    nuclear_mpm::viz::OrbitCamera horizontal_camera;
    const Eigen::Vector3f horizontal_feature{0.5F, 0.5F, 0.75F};
    const auto horizontal_before = project_to_ndc(horizontal_camera, horizontal_feature);
    const auto horizontal_input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 0.0F}, true, false, 0.0F);
    horizontal_camera.orbit(Eigen::Vector2f{horizontal_input.orbit_x, horizontal_input.orbit_y});
    const auto horizontal_after = project_to_ndc(horizontal_camera, horizontal_feature);

    REQUIRE(horizontal_after.x() > horizontal_before.x());

    nuclear_mpm::viz::OrbitCamera vertical_camera;
    const Eigen::Vector3f vertical_feature{0.5F, 0.75F, 0.5F};
    const auto vertical_before = project_to_ndc(vertical_camera, vertical_feature);
    const auto vertical_input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{0.0F, 10.0F}, true, false, 0.0F);
    vertical_camera.orbit(Eigen::Vector2f{vertical_input.orbit_x, vertical_input.orbit_y});
    const auto vertical_after = project_to_ndc(vertical_camera, vertical_feature);

    REQUIRE(vertical_after.y() < vertical_before.y());
}

TEST_CASE("mapped pan drag moves the projected scene with the cursor", "[camera]") {
    nuclear_mpm::viz::OrbitCamera camera;
    const Eigen::Vector3f scene_center = Eigen::Vector3f::Constant(0.5F);
    const auto before = project_to_ndc(camera, scene_center);
    const auto input = nuclear_mpm::viz::map_pointer_camera_input(
            Eigen::Vector2f{20.0F, 10.0F}, false, true, 0.0F);
    camera.pan(Eigen::Vector2f{input.pan_x, input.pan_y});
    const auto after = project_to_ndc(camera, scene_center);

    REQUIRE(after.x() > before.x());
    REQUIRE(after.y() < before.y());
}

TEST_CASE("UI mouse capture neutralizes camera input", "[camera]") {
    const nuclear_mpm::viz::CameraInput input{0.1F, 0.2F, 0.3F, 0.4F, 1.0F};

    const auto captured = nuclear_mpm::viz::filter_camera_input(input, true);
    const auto active = nuclear_mpm::viz::filter_camera_input(input, false);

    REQUIRE(captured == nuclear_mpm::viz::CameraInput{});
    REQUIRE(active == input);
}
