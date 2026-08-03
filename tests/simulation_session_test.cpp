#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "simulation_session.h"

using Catch::Approx;

TEST_CASE("switching dimensions preserves both simulation sessions", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    sessions.set_active_paused(false);
    const auto initial_2d_position = sessions.simulation_2d().particles().front().position;

    sessions.advance_active(1.0e-4F);
    const auto advanced_2d_position = sessions.simulation_2d().particles().front().position;
    const auto time_2d = sessions.simulation_2d().simulation_time();

    sessions.switch_dimension(nuclear_mpm::RuntimeDimension::Three);
    sessions.set_active_paused(false);
    sessions.advance_active(1.0e-4F);
    const auto advanced_3d_position = sessions.simulation_3d().particles().front().position;
    const auto time_3d = sessions.simulation_3d().simulation_time();

    sessions.switch_dimension(nuclear_mpm::RuntimeDimension::Two);

    REQUIRE(time_2d > 0.0F);
    REQUIRE(time_3d > 0.0F);
    REQUIRE((advanced_2d_position - initial_2d_position).norm() > 0.0F);
    REQUIRE(sessions.simulation_2d().simulation_time() == Approx(time_2d));
    REQUIRE(sessions.simulation_3d().simulation_time() == Approx(time_3d));
    REQUIRE((sessions.simulation_3d().particles().front().position - advanced_3d_position).norm() < 1.0e-7F);
}

TEST_CASE("staged topology applies only when active session resets", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    const int original_cells = sessions.simulation_2d().config().cells_per_axis;

    sessions.staged_2d().cells_per_axis = original_cells + 2;

    REQUIRE(sessions.simulation_2d().config().cells_per_axis == original_cells);
    sessions.reset_active();
    REQUIRE(sessions.simulation_2d().config().cells_per_axis == original_cells + 2);
    REQUIRE(sessions.simulation_2d().simulation_time() == 0.0F);
}

TEST_CASE("paused session advances only through single-step control", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    sessions.set_active_paused(true);

    sessions.advance_active(1.0e-4F);
    REQUIRE(sessions.simulation_2d().simulation_time() == 0.0F);

    sessions.single_step_active();
    REQUIRE(sessions.simulation_2d().simulation_time() > 0.0F);
}

TEST_CASE("live gravity updates only the active session", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    sessions.set_active_gravity(Eigen::Vector3f{1.0F, -2.0F, 3.0F});

    REQUIRE(sessions.simulation_2d().runtime_controls().gravity == nuclear_mpm::Vector<2>{1.0F, -2.0F});
    REQUIRE(sessions.simulation_3d().runtime_controls().gravity != nuclear_mpm::Vector<3>{1.0F, -2.0F, 3.0F});

    sessions.switch_dimension(nuclear_mpm::RuntimeDimension::Three);
    sessions.set_active_gravity(Eigen::Vector3f{1.0F, -2.0F, 3.0F});
    REQUIRE(sessions.simulation_3d().runtime_controls().gravity == nuclear_mpm::Vector<3>{1.0F, -2.0F, 3.0F});
}

TEST_CASE("render scenes use flat 2D and native 3D positions", "[session]") {
    nuclear_mpm::viz::SimulationSessionController sessions;
    const auto scene_2d = sessions.active_render_scene();
    REQUIRE(scene_2d.dimension == nuclear_mpm::RuntimeDimension::Two);
    REQUIRE_FALSE(scene_2d.points.empty());
    REQUIRE(scene_2d.points.front().z == Approx(0.0F));

    sessions.switch_dimension(nuclear_mpm::RuntimeDimension::Three);
    const auto scene_3d = sessions.active_render_scene();
    REQUIRE(scene_3d.dimension == nuclear_mpm::RuntimeDimension::Three);
    REQUIRE_FALSE(scene_3d.points.empty());
    REQUIRE(scene_3d.points.front().z > 0.0F);
}
