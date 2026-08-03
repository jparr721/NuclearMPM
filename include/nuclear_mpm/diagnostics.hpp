#pragma once

#include "types.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace nuclear_mpm {

enum class SimulationPhase {
    Initialization,
    ParticleToGrid,
    GridUpdate,
    GridToParticle,
    Advection,
};

[[nodiscard]] inline auto simulation_phase_name(SimulationPhase phase) -> const char * {
    switch (phase) {
        case SimulationPhase::Initialization:
            return "initialization";
        case SimulationPhase::ParticleToGrid:
            return "particle-to-grid";
        case SimulationPhase::GridUpdate:
            return "grid-update";
        case SimulationPhase::GridToParticle:
            return "grid-to-particle";
        case SimulationPhase::Advection:
            return "advection";
    }
    return "unknown";
}

class SimulationError : public std::runtime_error {
  public:
    SimulationError(SimulationPhase phase, std::string message,
                    std::optional<std::size_t> entity_index = std::nullopt)
        : std::runtime_error(std::string(simulation_phase_name(phase)) + ": " + std::move(message)), phase_(phase),
          entity_index_(entity_index) {}

    [[nodiscard]] auto phase() const noexcept -> SimulationPhase { return phase_; }
    [[nodiscard]] auto entity_index() const noexcept -> const std::optional<std::size_t> & { return entity_index_; }

  private:
    SimulationPhase phase_;
    std::optional<std::size_t> entity_index_;
};

template<int Dim>
struct StepDiagnostics {
    std::size_t substep_count = 0;
    Scalar minimum_time_step = Scalar{0};
    Scalar maximum_time_step = Scalar{0};
    Scalar particle_mass = Scalar{0};
    Scalar grid_mass = Scalar{0};
    Vector<Dim> linear_momentum = Vector<Dim>::Zero();
    Scalar maximum_particle_speed = Scalar{0};
    Scalar maximum_grid_speed = Scalar{0};
    Scalar minimum_deformation_determinant = Scalar{1};
    Scalar maximum_deformation_determinant = Scalar{1};
    Vector<Dim> particle_bounds_min = Vector<Dim>::Zero();
    Vector<Dim> particle_bounds_max = Vector<Dim>::Zero();
    std::size_t removed_outflow_particles = 0;
    Scalar removed_outflow_mass = Scalar{0};
};

} // namespace nuclear_mpm
