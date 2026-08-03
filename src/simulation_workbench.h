#pragma once

#include "simulation_session.h"
#include "visualizer.h"

#include <functional>
#include <string>

namespace nuclear_mpm::viz {

class SimulationWorkbench {
  public:
    SimulationWorkbench();
    ~SimulationWorkbench();

    SimulationWorkbench(const SimulationWorkbench &) = delete;
    auto operator=(const SimulationWorkbench &) -> SimulationWorkbench & = delete;

    auto run() -> void;

  private:
    auto draw_controls() -> void;
    auto draw_scene_settings() -> void;
    auto draw_boundary_settings() -> void;
    auto draw_diagnostics() -> void;
    auto perform_safely(const std::function<void()> &operation) -> void;

    Visualizer visualizer_;
    SimulationSessionController sessions_;
    Scalar simulation_seconds_per_frame_ = 1.0e-3F;
    std::string last_error_;
};

} // namespace nuclear_mpm::viz
