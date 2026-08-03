#pragma once

#include "boundary.hpp"
#include "material.hpp"

#include <cmath>
#include <stdexcept>

namespace nuclear_mpm {

template<int Dim>
struct SimulationConfig {
    int cells_per_axis = 64;
    Scalar max_time_step = Scalar{1.0e-4};
    Scalar cfl_factor = Scalar{0.4};
    MaterialTable<Dim> materials;
    DomainBoundaries<Dim> boundaries;
};

template<int Dim>
struct RuntimeControls {
    Vector<Dim> gravity = Vector<Dim>::Zero();
};

template<int Dim>
auto validate_runtime_controls(const RuntimeControls<Dim> &controls) -> void {
    if (!controls.gravity.allFinite()) { throw std::invalid_argument("gravity must be finite"); }
}

template<int Dim>
auto validate_simulation_config(const SimulationConfig<Dim> &config) -> void {
    if (config.cells_per_axis <= 0) { throw std::invalid_argument("cells_per_axis must be positive"); }
    if (!std::isfinite(config.max_time_step) || config.max_time_step <= Scalar{0}) {
        throw std::invalid_argument("max_time_step must be finite and positive");
    }
    if (!std::isfinite(config.cfl_factor) || config.cfl_factor <= Scalar{0} || config.cfl_factor > Scalar{1}) {
        throw std::invalid_argument("cfl_factor must be finite and in (0, 1]");
    }
    if (config.materials.empty()) { throw std::invalid_argument("at least one material must be registered"); }
    if (config.boundaries.boundary_layers <= 0 ||
        config.boundaries.boundary_layers * 2 > config.cells_per_axis) {
        throw std::invalid_argument("boundary_layers must fit inside the simulation domain");
    }
    if (config.boundaries.outflow_deletion_layers <= 0 ||
        config.boundaries.outflow_deletion_layers > config.boundaries.boundary_layers) {
        throw std::invalid_argument("outflow_deletion_layers must be positive and no wider than boundary_layers");
    }
}

} // namespace nuclear_mpm
