#pragma once

#include "config.hpp"
#include "diagnostics.hpp"
#include "stencil.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuclear_mpm {

template<int Dim>
class Simulation {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM simulations support only 2D and 3D");

  public:
    Simulation(SimulationConfig<Dim> config, RuntimeControls<Dim> runtime_controls,
               std::vector<Particle<Dim>> particles)
        : config_(validated_config(std::move(config))), runtime_controls_(validated_controls(runtime_controls)),
          grid_(config_.cells_per_axis), particles_(std::move(particles)),
          cell_size_(Scalar{1} / static_cast<Scalar>(config_.cells_per_axis)),
          inverse_cell_size_(static_cast<Scalar>(config_.cells_per_axis)) {
        validate_initial_particles();
    }

    [[nodiscard]] auto particles() const noexcept -> const std::vector<Particle<Dim>> & { return particles_; }
    [[nodiscard]] auto grid() const noexcept -> const Grid<Dim> & { return grid_; }
    [[nodiscard]] auto config() const noexcept -> const SimulationConfig<Dim> & { return config_; }
    [[nodiscard]] auto runtime_controls() const noexcept -> const RuntimeControls<Dim> & { return runtime_controls_; }
    [[nodiscard]] auto simulation_time() const noexcept -> Scalar { return simulation_time_; }

    auto set_runtime_controls(RuntimeControls<Dim> controls) -> void {
        validate_runtime_controls(controls);
        runtime_controls_ = controls;
    }

    [[nodiscard]] auto stable_time_step() const -> Scalar {
        Scalar maximum_transport_speed = Scalar{0};
        Scalar maximum_wave_speed = Scalar{0};
        for (const auto &particle : particles_) {
            maximum_transport_speed = std::max(maximum_transport_speed, particle.velocity.norm());
            const Scalar density = particle.mass / particle.reference_volume;
            maximum_wave_speed =
                    std::max(maximum_wave_speed,
                             characteristic_wave_speed(config_.materials.at(particle.material_id), density));
        }

        const Scalar characteristic_speed = maximum_transport_speed + maximum_wave_speed;
        if (characteristic_speed <= std::numeric_limits<Scalar>::epsilon()) { return config_.max_time_step; }

        const Scalar cfl_time_step = config_.cfl_factor * cell_size_ / characteristic_speed;
        if (!std::isfinite(cfl_time_step) || cfl_time_step <= Scalar{0}) {
            throw SimulationError(SimulationPhase::Initialization, "CFL estimate is non-finite or non-positive");
        }
        return std::min(config_.max_time_step, cfl_time_step);
    }

    auto advance(Scalar duration) -> StepDiagnostics<Dim> {
        if (!std::isfinite(duration) || duration <= Scalar{0}) {
            throw std::invalid_argument("advance duration must be finite and positive");
        }

        StepDiagnostics<Dim> aggregate;
        bool has_diagnostics = false;
        auto pending_removal = remove_outflow_particles();
        Scalar advanced = Scalar{0};

        while (advanced < duration) {
            if (aggregate.substep_count >= config_.max_substeps_per_advance) {
                throw SimulationError(SimulationPhase::Initialization,
                                      "advance exceeded max_substeps_per_advance");
            }

            const Scalar remaining = duration - advanced;
            const Scalar time_step = std::min(stable_time_step(), remaining);
            auto current = substep(time_step);
            advanced += time_step;

            const auto post_step_removal = remove_outflow_particles();
            current.removed_outflow_particles = pending_removal.first + post_step_removal.first;
            current.removed_outflow_mass = pending_removal.second + post_step_removal.second;
            pending_removal = {0, Scalar{0}};

            if (!has_diagnostics) {
                aggregate = current;
                has_diagnostics = true;
            } else {
                const auto previous_substeps = aggregate.substep_count;
                const auto previous_removed_particles = aggregate.removed_outflow_particles;
                const auto previous_removed_mass = aggregate.removed_outflow_mass;
                const auto minimum_time_step = std::min(aggregate.minimum_time_step, current.minimum_time_step);
                const auto maximum_time_step = std::max(aggregate.maximum_time_step, current.maximum_time_step);
                aggregate = current;
                aggregate.substep_count += previous_substeps;
                aggregate.removed_outflow_particles += previous_removed_particles;
                aggregate.removed_outflow_mass += previous_removed_mass;
                aggregate.minimum_time_step = minimum_time_step;
                aggregate.maximum_time_step = maximum_time_step;
            }

            const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * std::max(duration, Scalar{1});
            if (duration - advanced <= tolerance) { break; }
        }
        return aggregate;
    }

    auto substep(Scalar time_step) -> StepDiagnostics<Dim> {
        if (!std::isfinite(time_step) || time_step <= Scalar{0} || time_step > config_.max_time_step) {
            throw std::invalid_argument("substep time_step must be finite, positive, and no larger than max_time_step");
        }

        grid_.reset();
        particle_to_grid(time_step);
        update_grid(time_step);
        grid_to_particle(time_step);
        simulation_time_ += time_step;
        return collect_diagnostics(time_step);
    }

  private:
    [[nodiscard]] static auto validated_config(SimulationConfig<Dim> config) -> SimulationConfig<Dim> {
        validate_simulation_config(config);
        return config;
    }

    [[nodiscard]] static auto validated_controls(RuntimeControls<Dim> controls) -> RuntimeControls<Dim> {
        validate_runtime_controls(controls);
        return controls;
    }

    auto validate_initial_particles() -> void {
        for (std::size_t index = 0; index < particles_.size(); ++index) {
            try {
                validate_particle(particles_[index], index);
                (void) config_.materials.at(particles_[index].material_id);
            } catch (const std::out_of_range &error) {
                throw std::invalid_argument(std::string("particle ") + std::to_string(index) + ": " + error.what());
            }

            try {
                (void) QuadraticStencil<Dim>(particles_[index].position, inverse_cell_size_,
                                              grid_.nodes_per_axis());
            } catch (const std::exception &error) {
                throw SimulationError(SimulationPhase::Initialization, error.what(), index);
            }
        }
    }

    auto particle_to_grid(Scalar time_step) -> void {
        for (std::size_t index = 0; index < particles_.size(); ++index) {
            const auto &particle = particles_[index];
            try {
                const auto &material = config_.materials.at(particle.material_id);
                const Matrix<Dim> stress_impulse =
                        -time_step * particle.reference_volume * Scalar{4} * inverse_cell_size_ *
                        inverse_cell_size_ * kirchhoff_stress(material, particle);
                const Matrix<Dim> affine_momentum = stress_impulse + particle.mass * particle.affine_velocity;
                const QuadraticStencil<Dim> stencil(particle.position, inverse_cell_size_, grid_.nodes_per_axis());

                stencil.for_each_node([&](const StencilNode<Dim> &stencil_node) {
                    auto &grid_node = grid_.node(stencil_node.index);
                    grid_node.mass += stencil_node.weight * particle.mass;
                    grid_node.momentum +=
                            stencil_node.weight *
                            (particle.mass * particle.velocity + affine_momentum * stencil_node.physical_displacement);
                });
            } catch (const std::exception &error) {
                throw SimulationError(SimulationPhase::ParticleToGrid, error.what(), index);
            }
        }
    }

    auto update_grid(Scalar time_step) -> void {
        grid_.for_each_node([&](const Index<Dim> &, GridNode<Dim> &node) {
            if (node.mass > Scalar{0}) {
                node.velocity = node.momentum / node.mass + time_step * runtime_controls_.gravity;
            } else {
                node.velocity.setZero();
            }
        });
        apply_grid_boundaries(grid_, config_.boundaries);
    }

    auto grid_to_particle(Scalar time_step) -> void {
        for (std::size_t index = 0; index < particles_.size(); ++index) {
            auto &particle = particles_[index];
            try {
                const QuadraticStencil<Dim> stencil(particle.position, inverse_cell_size_, grid_.nodes_per_axis());
                Vector<Dim> new_velocity = Vector<Dim>::Zero();
                Matrix<Dim> new_affine_velocity = Matrix<Dim>::Zero();

                stencil.for_each_node([&](const StencilNode<Dim> &stencil_node) {
                    const auto &grid_velocity = grid_.node(stencil_node.index).velocity;
                    new_velocity += stencil_node.weight * grid_velocity;
                    new_affine_velocity += Scalar{4} * inverse_cell_size_ * inverse_cell_size_ *
                                           stencil_node.weight * grid_velocity *
                                           stencil_node.physical_displacement.transpose();
                });

                particle.velocity = new_velocity;
                particle.affine_velocity = new_affine_velocity;
                particle.position += time_step * particle.velocity;
                const Matrix<Dim> trial_deformation =
                        (scaled_identity<Dim>(Scalar{1}) + time_step * particle.affine_velocity) *
                        particle.elastic_deformation;
                particle.elastic_deformation =
                        project_deformation(config_.materials.at(particle.material_id), trial_deformation,
                                            particle.plastic_volume_ratio);
                validate_particle(particle, index);
            } catch (const std::exception &error) {
                throw SimulationError(SimulationPhase::GridToParticle, error.what(), index);
            }
        }
    }

    [[nodiscard]] auto remove_outflow_particles() -> std::pair<std::size_t, Scalar> {
        std::size_t removed_count = 0;
        Scalar removed_mass = Scalar{0};
        const Scalar lower_plane = static_cast<Scalar>(config_.boundaries.outflow_deletion_layers) * cell_size_;
        const Scalar upper_plane = Scalar{1} - lower_plane;

        const auto first_removed = std::remove_if(
                particles_.begin(), particles_.end(), [&](const Particle<Dim> &particle) {
                    for (int axis = 0; axis < Dim; ++axis) {
                        const bool leaves_lower = config_.boundaries.lower[axis] == BoundaryMode::Outflow &&
                                                  particle.position(axis) < lower_plane;
                        const bool leaves_upper = config_.boundaries.upper[axis] == BoundaryMode::Outflow &&
                                                  particle.position(axis) > upper_plane;
                        if (leaves_lower || leaves_upper) {
                            ++removed_count;
                            removed_mass += particle.mass;
                            return true;
                        }
                    }
                    return false;
                });
        particles_.erase(first_removed, particles_.end());
        return {removed_count, removed_mass};
    }

    [[nodiscard]] auto collect_diagnostics(Scalar time_step) const -> StepDiagnostics<Dim> {
        StepDiagnostics<Dim> diagnostics;
        diagnostics.substep_count = 1;
        diagnostics.minimum_time_step = time_step;
        diagnostics.maximum_time_step = time_step;

        if (!particles_.empty()) {
            diagnostics.particle_bounds_min = particles_.front().position;
            diagnostics.particle_bounds_max = particles_.front().position;
            diagnostics.minimum_deformation_determinant =
                    particles_.front().elastic_deformation.determinant();
            diagnostics.maximum_deformation_determinant = diagnostics.minimum_deformation_determinant;
        }

        for (const auto &particle : particles_) {
            diagnostics.particle_mass += particle.mass;
            diagnostics.linear_momentum += particle.mass * particle.velocity;
            diagnostics.maximum_particle_speed = std::max(diagnostics.maximum_particle_speed, particle.velocity.norm());
            diagnostics.particle_bounds_min = diagnostics.particle_bounds_min.cwiseMin(particle.position);
            diagnostics.particle_bounds_max = diagnostics.particle_bounds_max.cwiseMax(particle.position);
            const Scalar determinant = particle.elastic_deformation.determinant();
            diagnostics.minimum_deformation_determinant =
                    std::min(diagnostics.minimum_deformation_determinant, determinant);
            diagnostics.maximum_deformation_determinant =
                    std::max(diagnostics.maximum_deformation_determinant, determinant);
        }

        grid_.for_each_node([&](const Index<Dim> &, const GridNode<Dim> &node) {
            diagnostics.grid_mass += node.mass;
            diagnostics.maximum_grid_speed = std::max(diagnostics.maximum_grid_speed, node.velocity.norm());
        });
        return diagnostics;
    }

    SimulationConfig<Dim> config_;
    RuntimeControls<Dim> runtime_controls_;
    Grid<Dim> grid_;
    std::vector<Particle<Dim>> particles_;
    Scalar cell_size_;
    Scalar inverse_cell_size_;
    Scalar simulation_time_ = Scalar{0};
};

} // namespace nuclear_mpm
