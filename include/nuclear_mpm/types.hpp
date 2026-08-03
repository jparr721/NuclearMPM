#pragma once

#include <Eigen/Core>

namespace nuclear_mpm {

using Scalar = float;

template<int Dim>
using Vector = Eigen::Matrix<Scalar, Dim, 1>;

template<int Dim>
using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;

template<int Dim>
using Index = Eigen::Matrix<int, Dim, 1>;

enum class RuntimeDimension : int {
    Two = 2,
    Three = 3,
};

template<int Dim>
inline constexpr bool is_supported_dimension = Dim == 2 || Dim == 3;

} // namespace nuclear_mpm
