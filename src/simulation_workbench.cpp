#include "simulation_workbench.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace nuclear_mpm::viz {
namespace {

constexpr std::array<const char *, 3> kMaterialNames{"Solid", "Snow", "Liquid"};
constexpr std::array<const char *, 3> kBoundaryNames{"Slip", "Sticky", "Outflow"};
constexpr std::array<const char *, 3> kAxisNames{"X", "Y", "Z"};

template<int Dim>
auto edit_vector(const char *label, Vector<Dim> &value, float speed, float minimum, float maximum) -> bool {
    if constexpr (Dim == 2) {
        return ImGui::DragFloat2(label, value.data(), speed, minimum, maximum);
    } else {
        return ImGui::DragFloat3(label, value.data(), speed, minimum, maximum);
    }
}

template<int Dim>
auto edit_index(const char *label, Index<Dim> &value, int minimum, int maximum) -> bool {
    bool changed = false;
    if constexpr (Dim == 2) {
        changed = ImGui::InputInt2(label, value.data());
    } else {
        changed = ImGui::InputInt3(label, value.data());
    }
    if (changed) {
        for (int axis = 0; axis < Dim; ++axis) { value(axis) = std::clamp(value(axis), minimum, maximum); }
    }
    return changed;
}

template<int Dim>
auto draw_blueprint(SessionBlueprint<Dim> &blueprint) -> void {
    ImGui::TextUnformatted("Staged scene (applies on reset)");
    ImGui::InputInt("Grid cells", &blueprint.cells_per_axis);
    edit_index("Points per axis", blueprint.points_per_axis, 1, 64);
    edit_vector("Box minimum", blueprint.box_min, 0.005F, 0.01F, 0.99F);
    edit_vector("Box maximum", blueprint.box_max, 0.005F, 0.01F, 0.99F);
    ImGui::DragFloat("Density", &blueprint.density, 0.01F, 0.01F, 100.0F);
    ImGui::DragFloat("Maximum dt", &blueprint.max_time_step, 1.0e-5F, 1.0e-6F, 1.0e-2F, "%.6g");
    ImGui::DragFloat("CFL factor", &blueprint.cfl_factor, 0.01F, 0.01F, 1.0F);

    int material = static_cast<int>(blueprint.material);
    if (ImGui::Combo("Material", &material, kMaterialNames.data(), static_cast<int>(kMaterialNames.size()))) {
        blueprint.material = static_cast<MaterialPreset>(material);
    }
}

template<int Dim>
auto draw_boundaries(SessionBlueprint<Dim> &blueprint) -> void {
    ImGui::InputInt("Boundary layers", &blueprint.boundaries.boundary_layers);
    ImGui::InputInt("Outflow deletion layers", &blueprint.boundaries.outflow_deletion_layers);
    for (int axis = 0; axis < Dim; ++axis) {
        ImGui::PushID(axis);
        int lower = static_cast<int>(blueprint.boundaries.lower[axis]);
        int upper = static_cast<int>(blueprint.boundaries.upper[axis]);
        const std::string lower_label = std::string("Lower ") + kAxisNames[static_cast<std::size_t>(axis)];
        const std::string upper_label = std::string("Upper ") + kAxisNames[static_cast<std::size_t>(axis)];
        if (ImGui::Combo(lower_label.c_str(), &lower, kBoundaryNames.data(),
                         static_cast<int>(kBoundaryNames.size()))) {
            blueprint.boundaries.lower[axis] = static_cast<BoundaryMode>(lower);
        }
        if (ImGui::Combo(upper_label.c_str(), &upper, kBoundaryNames.data(),
                         static_cast<int>(kBoundaryNames.size()))) {
            blueprint.boundaries.upper[axis] = static_cast<BoundaryMode>(upper);
        }
        ImGui::PopID();
    }
}

template<int Dim>
auto draw_diagnostics_values(const Simulation<Dim> &simulation, const StepDiagnostics<Dim> &diagnostics) -> void {
    ImGui::Text("Particles: %zu", simulation.particles().size());
    ImGui::Text("Simulation time: %.6f", simulation.simulation_time());
    ImGui::Text("Last substeps: %zu", diagnostics.substep_count);
    ImGui::Text("Particle mass: %.6g", diagnostics.particle_mass);
    ImGui::Text("Grid mass: %.6g", diagnostics.grid_mass);
    ImGui::Text("Maximum particle speed: %.6g", diagnostics.maximum_particle_speed);
    ImGui::Text("Maximum grid speed: %.6g", diagnostics.maximum_grid_speed);
    ImGui::Text("det(F): [%.6g, %.6g]", diagnostics.minimum_deformation_determinant,
                diagnostics.maximum_deformation_determinant);
    ImGui::Text("Outflow removals: %zu (mass %.6g)", diagnostics.removed_outflow_particles,
                diagnostics.removed_outflow_mass);
}

} // namespace

SimulationWorkbench::SimulationWorkbench() : visualizer_("NuclearMPM Workbench", 1200, 800) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    if (!ImGui_ImplGlfw_InitForOpenGL(visualizer_.native_window(), true)) {
        ImGui::DestroyContext();
        throw std::runtime_error("Dear ImGui GLFW backend initialization failed");
    }
    if (!ImGui_ImplOpenGL3_Init("#version 410 core")) {
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        throw std::runtime_error("Dear ImGui OpenGL backend initialization failed");
    }
}

SimulationWorkbench::~SimulationWorkbench() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

auto SimulationWorkbench::perform_safely(const std::function<void()> &operation) -> void {
    try {
        operation();
    } catch (const std::exception &error) {
        last_error_ = error.what();
        sessions_.set_active_paused(true);
    }
}

auto SimulationWorkbench::draw_scene_settings() -> void {
    if (sessions_.active_dimension() == RuntimeDimension::Two) {
        auto &blueprint = sessions_.staged_2d();
        draw_blueprint(blueprint);
        if (edit_vector("Gravity", blueprint.gravity, 0.1F, -100.0F, 100.0F)) {
            sessions_.set_active_gravity(Eigen::Vector3f{blueprint.gravity(0), blueprint.gravity(1), 0.0F});
        }
    } else {
        auto &blueprint = sessions_.staged_3d();
        draw_blueprint(blueprint);
        if (edit_vector("Gravity", blueprint.gravity, 0.1F, -100.0F, 100.0F)) {
            sessions_.set_active_gravity(blueprint.gravity);
        }
    }
}

auto SimulationWorkbench::draw_boundary_settings() -> void {
    if (sessions_.active_dimension() == RuntimeDimension::Two) {
        draw_boundaries(sessions_.staged_2d());
    } else {
        draw_boundaries(sessions_.staged_3d());
    }
}

auto SimulationWorkbench::draw_diagnostics() -> void {
    if (sessions_.active_dimension() == RuntimeDimension::Two) {
        draw_diagnostics_values(sessions_.simulation_2d(), sessions_.diagnostics_2d());
    } else {
        draw_diagnostics_values(sessions_.simulation_3d(), sessions_.diagnostics_3d());
    }
}

auto SimulationWorkbench::draw_controls() -> void {
    ImGui::SetNextWindowSize(ImVec2(370.0F, 720.0F), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation Workbench");

    int dimension = sessions_.active_dimension() == RuntimeDimension::Two ? 2 : 3;
    if (ImGui::RadioButton("2D", dimension == 2)) {
        sessions_.switch_dimension(RuntimeDimension::Two);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("3D", dimension == 3)) {
        sessions_.switch_dimension(RuntimeDimension::Three);
    }

    bool paused = sessions_.active_paused();
    if (ImGui::Checkbox("Paused", &paused)) { sessions_.set_active_paused(paused); }
    ImGui::SameLine();
    if (ImGui::Button("Single step")) { perform_safely([&] { sessions_.single_step_active(); }); }
    if (ImGui::Button("Apply staged settings / reset")) { perform_safely([&] { sessions_.reset_active(); }); }
    ImGui::SameLine();
    if (ImGui::Button("Reset camera")) { sessions_.reset_active_camera(); }
    ImGui::DragFloat("Simulation seconds/frame", &simulation_seconds_per_frame_, 1.0e-4F, 1.0e-5F, 1.0e-2F,
                     "%.6g");

    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) { draw_scene_settings(); }
    if (ImGui::CollapsingHeader("Boundaries")) { draw_boundary_settings(); }
    if (ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen)) { draw_diagnostics(); }
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (sessions_.active_dimension() == RuntimeDimension::Two) {
            ImGui::TextUnformatted("Middle drag: pan\nScroll: zoom");
        } else {
            ImGui::TextUnformatted("Left drag: orbit\nMiddle drag: pan\nScroll: dolly");
        }
    }

    if (!last_error_.empty()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0F, 0.35F, 0.35F, 1.0F), "Simulation paused: %s", last_error_.c_str());
        if (ImGui::Button("Clear error")) { last_error_.clear(); }
    }
    ImGui::End();
}

auto SimulationWorkbench::run() -> void {
    while (!visualizer_.should_close()) {
        visualizer_.begin_frame();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        draw_controls();
        const auto input = filter_camera_input(visualizer_.camera_input(), ImGui::GetIO().WantCaptureMouse);
        sessions_.apply_camera_input(input);
        if (!sessions_.active_paused()) {
            perform_safely([&] { sessions_.advance_active(simulation_seconds_per_frame_); });
        }

        const auto scene = sessions_.active_render_scene();
        const auto view_projection = sessions_.active_view_projection(visualizer_.framebuffer_aspect_ratio());
        visualizer_.draw_scene(scene, view_projection);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        visualizer_.end_frame();
    }
}

} // namespace nuclear_mpm::viz
