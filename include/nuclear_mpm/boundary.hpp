#pragma once

#include "grid.hpp"

#include <array>
#include <stdexcept>

namespace nuclear_mpm {

enum class BoundaryMode {
    Slip,
    Sticky,
    Outflow,
};

template<int Dim>
struct DomainBoundaries {
    DomainBoundaries() {
        lower.fill(BoundaryMode::Slip);
        upper.fill(BoundaryMode::Slip);
    }

    std::array<BoundaryMode, Dim> lower;
    std::array<BoundaryMode, Dim> upper;
    int boundary_layers = 3;
    int outflow_deletion_layers = 2;
};

namespace detail {

template<int Dim>
auto apply_boundary_mode(BoundaryMode mode, int axis, GridNode<Dim> &node) -> void {
    switch (mode) {
        case BoundaryMode::Slip:
            node.velocity(axis) = Scalar{0};
            break;
        case BoundaryMode::Sticky:
            node.velocity.setZero();
            break;
        case BoundaryMode::Outflow:
            break;
    }
}

} // namespace detail

template<int Dim>
auto apply_grid_boundaries(Grid<Dim> &grid, const DomainBoundaries<Dim> &boundaries) -> void {
    if (boundaries.boundary_layers <= 0 || boundaries.boundary_layers >= grid.nodes_per_axis()) {
        throw std::invalid_argument("boundary_layers must fit inside the grid");
    }

    grid.for_each_node([&](const Index<Dim> &coordinate, GridNode<Dim> &node) {
        for (int axis = 0; axis < Dim; ++axis) {
            if (coordinate(axis) < boundaries.boundary_layers && node.velocity(axis) < Scalar{0}) {
                detail::apply_boundary_mode(boundaries.lower[axis], axis, node);
            }
            if (coordinate(axis) >= grid.nodes_per_axis() - boundaries.boundary_layers &&
                node.velocity(axis) > Scalar{0}) {
                detail::apply_boundary_mode(boundaries.upper[axis], axis, node);
            }
        }
    });
}

} // namespace nuclear_mpm
