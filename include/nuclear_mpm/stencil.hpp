#pragma once

#include "types.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace nuclear_mpm {

template<int Dim>
struct StencilNode {
    Index<Dim> index;
    Index<Dim> offset;
    Scalar weight;
    Vector<Dim> grid_displacement;
    Vector<Dim> physical_displacement;
};

template<int Dim>
class QuadraticStencil {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM stencils support only 2D and 3D");

  public:
    QuadraticStencil(const Vector<Dim> &position, Scalar inverse_cell_size, int nodes_per_axis)
        : inverse_cell_size_(inverse_cell_size), nodes_per_axis_(nodes_per_axis) {
        if (!position.allFinite()) { throw std::invalid_argument("stencil position must be finite"); }
        if (!std::isfinite(inverse_cell_size) || inverse_cell_size <= Scalar{0}) {
            throw std::invalid_argument("stencil inverse_cell_size must be finite and positive");
        }
        if (nodes_per_axis < 3) { throw std::invalid_argument("quadratic stencil requires at least three grid nodes"); }

        const Vector<Dim> grid_position = position * inverse_cell_size_;
        base_ = (grid_position.array() - Scalar{0.5}).floor().template cast<int>();
        fractional_ = grid_position - base_.template cast<Scalar>();

        for (int axis = 0; axis < Dim; ++axis) {
            if (base_(axis) < 0 || base_(axis) + 2 >= nodes_per_axis_) {
                throw std::out_of_range("particle quadratic stencil is outside the grid domain");
            }

            const Scalar fraction = fractional_(axis);
            weights_(axis, 0) = Scalar{0.5} * (Scalar{1.5} - fraction) * (Scalar{1.5} - fraction);
            weights_(axis, 1) = Scalar{0.75} - (fraction - Scalar{1}) * (fraction - Scalar{1});
            weights_(axis, 2) = Scalar{0.5} * (fraction - Scalar{0.5}) * (fraction - Scalar{0.5});
        }
    }

    [[nodiscard]] auto base() const noexcept -> const Index<Dim> & { return base_; }
    [[nodiscard]] auto fractional_position() const noexcept -> const Vector<Dim> & { return fractional_; }

    template<typename Callback>
    auto for_each_node(Callback &&callback) const -> void {
        constexpr std::size_t stencil_width = 3;
        constexpr std::size_t node_total = integer_power(stencil_width, Dim);

        for (std::size_t ordinal = 0; ordinal < node_total; ++ordinal) {
            auto remainder = ordinal;
            Index<Dim> offset;
            for (int axis = Dim - 1; axis >= 0; --axis) {
                offset(axis) = static_cast<int>(remainder % stencil_width);
                remainder /= stencil_width;
            }

            Scalar weight = Scalar{1};
            for (int axis = 0; axis < Dim; ++axis) { weight *= weights_(axis, offset(axis)); }

            const Vector<Dim> grid_displacement = offset.template cast<Scalar>() - fractional_;
            callback(StencilNode<Dim>{
                    base_ + offset,
                    offset,
                    weight,
                    grid_displacement,
                    grid_displacement / inverse_cell_size_,
            });
        }
    }

  private:
    [[nodiscard]] static constexpr auto integer_power(std::size_t base, int exponent) -> std::size_t {
        std::size_t result = 1;
        for (int value = 0; value < exponent; ++value) { result *= base; }
        return result;
    }

    Scalar inverse_cell_size_;
    int nodes_per_axis_;
    Index<Dim> base_ = Index<Dim>::Zero();
    Vector<Dim> fractional_ = Vector<Dim>::Zero();
    Eigen::Matrix<Scalar, Dim, 3> weights_ = Eigen::Matrix<Scalar, Dim, 3>::Zero();
};

} // namespace nuclear_mpm
