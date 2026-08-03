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

    auto set_runtime_controls(RuntimeControls<Dim> controls) -> void {
        validate_runtime_controls(controls);
        runtime_controls_ = controls;
    }

    auto substep(Scalar time_step) -> StepDiagnostics<Dim> {
        if (!std::isfinite(time_step) || time_step <= Scalar{0} || time_step > config_.max_time_step) {
            throw std::invalid_argument("substep time_step must be finite, positive, and no larger than max_time_step");
        }

        grid_.reset();
        particle_to_grid(time_step);
        update_grid(time_step);
        grid_to_particle(time_step);
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
};

} // namespace nuclear_mpm
