#pragma once

#include "types.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nuclear_mpm {

template<int Dim>
struct GridNode {
    Scalar mass = Scalar{0};
    Vector<Dim> momentum = Vector<Dim>::Zero();
    Vector<Dim> velocity = Vector<Dim>::Zero();
};

template<int Dim>
class Grid {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM grids support only 2D and 3D");

  public:
    explicit Grid(int cells_per_axis)
        : cells_per_axis_(cells_per_axis), nodes_per_axis_(cells_per_axis + 1), nodes_(compute_node_count(cells_per_axis)) {
        if (cells_per_axis <= 0) { throw std::invalid_argument("grid cells_per_axis must be positive"); }
    }

    [[nodiscard]] auto cells_per_axis() const noexcept -> int { return cells_per_axis_; }
    [[nodiscard]] auto nodes_per_axis() const noexcept -> int { return nodes_per_axis_; }
    [[nodiscard]] auto node_count() const noexcept -> std::size_t { return nodes_.size(); }

    [[nodiscard]] auto flatten(const Index<Dim> &coordinate_value) const -> std::size_t {
        std::size_t flat = 0;
        for (int axis = 0; axis < Dim; ++axis) {
            const int component = coordinate_value(axis);
            if (component < 0 || component >= nodes_per_axis_) {
                throw std::out_of_range("grid coordinate is outside the node domain");
            }
            flat = flat * static_cast<std::size_t>(nodes_per_axis_) + static_cast<std::size_t>(component);
        }
        return flat;
    }

    [[nodiscard]] auto coordinate(std::size_t flat) const -> Index<Dim> {
        if (flat >= nodes_.size()) { throw std::out_of_range("flat grid index is outside storage"); }

        Index<Dim> result;
        for (int axis = Dim - 1; axis >= 0; --axis) {
            result(axis) = static_cast<int>(flat % static_cast<std::size_t>(nodes_per_axis_));
            flat /= static_cast<std::size_t>(nodes_per_axis_);
        }
        return result;
    }

    auto node(const Index<Dim> &coordinate_value) -> GridNode<Dim> & { return nodes_.at(flatten(coordinate_value)); }
    [[nodiscard]] auto node(const Index<Dim> &coordinate_value) const -> const GridNode<Dim> & {
        return nodes_.at(flatten(coordinate_value));
    }

    auto reset() noexcept -> void {
        for (auto &node_value : nodes_) { node_value = GridNode<Dim>{}; }
    }

  private:
    [[nodiscard]] static auto compute_node_count(int cells_per_axis) -> std::size_t {
        if (cells_per_axis <= 0) { throw std::invalid_argument("grid cells_per_axis must be positive"); }

        const auto nodes_per_axis = static_cast<std::size_t>(cells_per_axis) + 1U;
        std::size_t result = 1;
        for (int axis = 0; axis < Dim; ++axis) {
            if (result > std::numeric_limits<std::size_t>::max() / nodes_per_axis) {
                throw std::overflow_error("grid node count exceeds addressable storage");
            }
            result *= nodes_per_axis;
        }
        return result;
    }

    int cells_per_axis_;
    int nodes_per_axis_;
    std::vector<GridNode<Dim>> nodes_;
};

} // namespace nuclear_mpm
