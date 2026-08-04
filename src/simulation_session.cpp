#include "simulation_session.h"

#include <stdexcept>
#include <utility>

namespace nuclear_mpm::viz {
namespace {

[[nodiscard]] auto preset_color(MaterialPreset preset) -> std::uint32_t {
    switch (preset) {
        case MaterialPreset::Solid:
            return 0xED553BU;
        case MaterialPreset::Snow:
            return 0xF2F2F2U;
        case MaterialPreset::Liquid:
            return 0x068587U;
    }
    return 0xED553BU;
}

template<int Dim>
auto add_material(MaterialTable<Dim> &materials, MaterialPreset preset) -> void {
    switch (preset) {
        case MaterialPreset::Solid:
            materials.add(FixedCorotatedSolid<Dim>(400.0F, 0.2F));
            break;
        case MaterialPreset::Snow:
            materials.add(Snow<Dim>(400.0F, 0.2F));
            break;
        case MaterialPreset::Liquid:
            materials.add(WeaklyCompressibleLiquid<Dim>(400.0F));
            break;
    }
}

template<int Dim>
auto populate_grid_overlay(RenderGridOverlay &overlay, const Simulation<Dim> &simulation) -> void {
    const auto &config = simulation.config();
    overlay.cells_per_axis = config.cells_per_axis;
    overlay.boundary_layers = config.boundaries.boundary_layers;
    overlay.outflow_deletion_layers = config.boundaries.outflow_deletion_layers;
    for (int axis = 0; axis < Dim; ++axis) {
        overlay.lower[static_cast<std::size_t>(axis)] = config.boundaries.lower[axis];
        overlay.upper[static_cast<std::size_t>(axis)] = config.boundaries.upper[axis];
    }
}

} // namespace

template<int Dim>
auto SimulationSessionController::create_simulation(const SessionBlueprint<Dim> &blueprint)
        -> std::unique_ptr<Simulation<Dim>> {
    SimulationConfig<Dim> config;
    config.cells_per_axis = blueprint.cells_per_axis;
    config.max_time_step = blueprint.max_time_step;
    config.cfl_factor = blueprint.cfl_factor;
    config.max_substeps_per_advance = 512;
    config.boundaries = blueprint.boundaries;
    add_material(config.materials, blueprint.material);

    RuntimeControls<Dim> controls;
    controls.gravity = blueprint.gravity;
    auto particles = sample_box_lattice<Dim>(blueprint.points_per_axis, blueprint.box_min, blueprint.box_max,
                                              blueprint.density, 0, preset_color(blueprint.material));
    return std::make_unique<Simulation<Dim>>(std::move(config), controls, std::move(particles));
}

SimulationSessionController::SimulationSessionController()
    : simulation_2d_(create_simulation(staged_2d_)), simulation_3d_(create_simulation(staged_3d_)) {}

auto SimulationSessionController::active_dimension() const noexcept -> RuntimeDimension { return active_dimension_; }

auto SimulationSessionController::switch_dimension(RuntimeDimension dimension) noexcept -> void {
    active_dimension_ = dimension;
}

auto SimulationSessionController::active_paused() const noexcept -> bool {
    return active_dimension_ == RuntimeDimension::Two ? paused_2d_ : paused_3d_;
}

auto SimulationSessionController::set_active_paused(bool paused) noexcept -> void {
    if (active_dimension_ == RuntimeDimension::Two) {
        paused_2d_ = paused;
    } else {
        paused_3d_ = paused;
    }
}

auto SimulationSessionController::set_active_gravity(const Eigen::Vector3f &gravity) -> void {
    if (!gravity.allFinite()) { throw std::invalid_argument("session gravity must be finite"); }
    if (active_dimension_ == RuntimeDimension::Two) {
        staged_2d_.gravity = gravity.head<2>();
        auto controls = simulation_2d_->runtime_controls();
        controls.gravity = staged_2d_.gravity;
        simulation_2d_->set_runtime_controls(controls);
    } else {
        staged_3d_.gravity = gravity;
        auto controls = simulation_3d_->runtime_controls();
        controls.gravity = staged_3d_.gravity;
        simulation_3d_->set_runtime_controls(controls);
    }
}

auto SimulationSessionController::advance_active(Scalar duration) -> void {
    if (active_paused()) { return; }
    if (active_dimension_ == RuntimeDimension::Two) {
        diagnostics_2d_ = simulation_2d_->advance(duration);
    } else {
        diagnostics_3d_ = simulation_3d_->advance(duration);
    }
}

auto SimulationSessionController::single_step_active() -> void {
    if (active_dimension_ == RuntimeDimension::Two) {
        diagnostics_2d_ = simulation_2d_->substep(simulation_2d_->stable_time_step());
    } else {
        diagnostics_3d_ = simulation_3d_->substep(simulation_3d_->stable_time_step());
    }
}

auto SimulationSessionController::reset_active() -> void {
    if (active_dimension_ == RuntimeDimension::Two) {
        simulation_2d_ = create_simulation(staged_2d_);
        diagnostics_2d_ = {};
        camera_2d_.reset(DomainBounds{});
    } else {
        simulation_3d_ = create_simulation(staged_3d_);
        diagnostics_3d_ = {};
        camera_3d_.reset(DomainBounds{});
    }
}

auto SimulationSessionController::simulation_2d() const noexcept -> const Simulation<2> & { return *simulation_2d_; }
auto SimulationSessionController::simulation_3d() const noexcept -> const Simulation<3> & { return *simulation_3d_; }
auto SimulationSessionController::staged_2d() noexcept -> SessionBlueprint<2> & { return staged_2d_; }
auto SimulationSessionController::staged_3d() noexcept -> SessionBlueprint<3> & { return staged_3d_; }
auto SimulationSessionController::diagnostics_2d() const noexcept -> const StepDiagnostics<2> & {
    return diagnostics_2d_;
}
auto SimulationSessionController::diagnostics_3d() const noexcept -> const StepDiagnostics<3> & {
    return diagnostics_3d_;
}

auto SimulationSessionController::active_render_scene() const -> RenderScene {
    RenderScene scene;
    scene.dimension = active_dimension_;
    if (active_dimension_ == RuntimeDimension::Two) {
        populate_grid_overlay(scene.grid_overlay, *simulation_2d_);
        scene.points.reserve(simulation_2d_->particles().size());
        for (const auto &particle : simulation_2d_->particles()) {
            scene.points.push_back({particle.position(0), particle.position(1), 0.0F, particle.user_tag});
        }
    } else {
        populate_grid_overlay(scene.grid_overlay, *simulation_3d_);
        scene.points.reserve(simulation_3d_->particles().size());
        for (const auto &particle : simulation_3d_->particles()) {
            scene.points.push_back(
                    {particle.position(0), particle.position(1), particle.position(2), particle.user_tag});
        }
    }
    return scene;
}

auto SimulationSessionController::apply_camera_input(const CameraInput &input) -> void {
    if (active_dimension_ == RuntimeDimension::Two) {
        camera_2d_.pan(Eigen::Vector2f{input.pan_x, input.pan_y});
        camera_2d_.zoom(input.dolly);
    } else {
        camera_3d_.orbit(Eigen::Vector2f{input.orbit_x, input.orbit_y});
        camera_3d_.pan(Eigen::Vector2f{input.pan_x, input.pan_y});
        camera_3d_.dolly(input.dolly);
    }
}

auto SimulationSessionController::reset_active_camera() -> void {
    if (active_dimension_ == RuntimeDimension::Two) {
        camera_2d_.reset(DomainBounds{});
    } else {
        camera_3d_.reset(DomainBounds{});
    }
}

auto SimulationSessionController::active_view_projection(float aspect_ratio) const -> Eigen::Matrix4f {
    if (active_dimension_ == RuntimeDimension::Two) {
        return camera_2d_.projection_matrix(aspect_ratio) * camera_2d_.view_matrix();
    }
    return camera_3d_.projection_matrix(aspect_ratio) * camera_3d_.view_matrix();
}

template auto SimulationSessionController::create_simulation<2>(const SessionBlueprint<2> &)
        -> std::unique_ptr<Simulation<2>>;
template auto SimulationSessionController::create_simulation<3>(const SessionBlueprint<3> &)
        -> std::unique_ptr<Simulation<3>>;

} // namespace nuclear_mpm::viz
