#pragma once

#include "visualizer.h"

#include <nuclear_mpm/nuclear_mpm.hpp>

#include <Eigen/Core>

#include <memory>

namespace nuclear_mpm::viz {

enum class MaterialPreset {
    Solid,
    Snow,
    Liquid,
};

template<int Dim>
struct SessionBlueprint {
    SessionBlueprint() {
        points_per_axis.setConstant(Dim == 2 ? 8 : 3);
        gravity.setZero();
        gravity(1) = -9.81F;
    }

    int cells_per_axis = 64;
    Scalar max_time_step = 1.0e-4F;
    Scalar cfl_factor = 0.4F;
    Scalar density = 1.0F;
    Index<Dim> points_per_axis = Index<Dim>::Zero();
    Vector<Dim> box_min = Vector<Dim>::Constant(0.35F);
    Vector<Dim> box_max = Vector<Dim>::Constant(0.65F);
    Vector<Dim> gravity = Vector<Dim>::Zero();
    MaterialPreset material = MaterialPreset::Solid;
    DomainBoundaries<Dim> boundaries;
};

class SimulationSessionController {
  public:
    SimulationSessionController();

    [[nodiscard]] auto active_dimension() const noexcept -> RuntimeDimension;
    auto switch_dimension(RuntimeDimension dimension) noexcept -> void;

    [[nodiscard]] auto active_paused() const noexcept -> bool;
    auto set_active_paused(bool paused) noexcept -> void;
    auto set_active_gravity(const Eigen::Vector3f &gravity) -> void;
    auto advance_active(Scalar duration) -> void;
    auto single_step_active() -> void;
    auto reset_active() -> void;

    [[nodiscard]] auto simulation_2d() const noexcept -> const Simulation<2> &;
    [[nodiscard]] auto simulation_3d() const noexcept -> const Simulation<3> &;
    [[nodiscard]] auto staged_2d() noexcept -> SessionBlueprint<2> &;
    [[nodiscard]] auto staged_3d() noexcept -> SessionBlueprint<3> &;
    [[nodiscard]] auto diagnostics_2d() const noexcept -> const StepDiagnostics<2> &;
    [[nodiscard]] auto diagnostics_3d() const noexcept -> const StepDiagnostics<3> &;

    [[nodiscard]] auto active_render_scene() const -> RenderScene;
    auto apply_camera_input(const CameraInput &input) -> void;
    auto reset_active_camera() -> void;
    [[nodiscard]] auto active_view_projection(float aspect_ratio) const -> Eigen::Matrix4f;

  private:
    template<int Dim>
    [[nodiscard]] static auto create_simulation(const SessionBlueprint<Dim> &blueprint)
            -> std::unique_ptr<Simulation<Dim>>;

    RuntimeDimension active_dimension_ = RuntimeDimension::Two;
    SessionBlueprint<2> staged_2d_;
    SessionBlueprint<3> staged_3d_;
    std::unique_ptr<Simulation<2>> simulation_2d_;
    std::unique_ptr<Simulation<3>> simulation_3d_;
    StepDiagnostics<2> diagnostics_2d_;
    StepDiagnostics<3> diagnostics_3d_;
    bool paused_2d_ = true;
    bool paused_3d_ = true;
    OrthographicCamera camera_2d_;
    OrbitCamera camera_3d_;
};

} // namespace nuclear_mpm::viz
