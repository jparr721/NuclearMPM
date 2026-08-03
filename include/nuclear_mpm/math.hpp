#pragma once

#include "types.hpp"

#include <Eigen/LU>
#include <Eigen/SVD>

#include <stdexcept>

namespace nuclear_mpm {

template<int Dim>
[[nodiscard]] auto scaled_identity(Scalar value) -> Matrix<Dim> {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM supports only 2D and 3D math");
    return Matrix<Dim>::Identity() * value;
}

template<int Dim>
[[nodiscard]] auto filled_matrix(Scalar value) -> Matrix<Dim> {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM supports only 2D and 3D math");
    return Matrix<Dim>::Constant(value);
}

template<int Dim>
struct SignedSvd {
    Matrix<Dim> u;
    Vector<Dim> singular_values;
    Matrix<Dim> v;
};

template<int Dim>
[[nodiscard]] auto signed_svd(const Matrix<Dim> &input) -> SignedSvd<Dim> {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM supports only 2D and 3D math");
    if (!input.allFinite()) { throw std::invalid_argument("signed_svd requires a finite matrix"); }

    const Eigen::JacobiSVD<Matrix<Dim>> decomposition(input, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto u = decomposition.matrixU();
    auto singular_values = decomposition.singularValues();
    auto v = decomposition.matrixV();
    constexpr int last = Dim - 1;

    if (u.determinant() < Scalar{0}) {
        u.col(last) *= Scalar{-1};
        singular_values(last) *= Scalar{-1};
    }
    if (v.determinant() < Scalar{0}) {
        v.col(last) *= Scalar{-1};
        singular_values(last) *= Scalar{-1};
    }

    return {u, singular_values, v};
}

template<int Dim>
struct PolarDecomposition {
    Matrix<Dim> rotation;
    Matrix<Dim> stretch;
};

template<int Dim>
[[nodiscard]] auto polar_decomposition(const Matrix<Dim> &input) -> PolarDecomposition<Dim> {
    const auto decomposition = signed_svd<Dim>(input);
    return {
            decomposition.u * decomposition.v.transpose(),
            decomposition.v * decomposition.singular_values.asDiagonal() * decomposition.v.transpose(),
    };
}

} // namespace nuclear_mpm
