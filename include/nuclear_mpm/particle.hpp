#pragma once

#include "math.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace nuclear_mpm {

template<int Dim>
struct Particle {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM particles support only 2D and 3D");

    Vector<Dim> position = Vector<Dim>::Zero();
    Vector<Dim> velocity = Vector<Dim>::Zero();
    Matrix<Dim> elastic_deformation = scaled_identity<Dim>(Scalar{1});
    Matrix<Dim> affine_velocity = Matrix<Dim>::Zero();
    Scalar plastic_volume_ratio = Scalar{1};
    Scalar mass = Scalar{0};
    Scalar reference_volume = Scalar{0};
    std::size_t material_id = 0;
    std::uint32_t user_tag = 0;
};

template<int Dim>
auto validate_particle(const Particle<Dim> &particle, std::size_t particle_index) -> void {
    const auto prefix = std::string("particle ") + std::to_string(particle_index) + ": ";
    if (!particle.position.allFinite() || !particle.velocity.allFinite() ||
        !particle.elastic_deformation.allFinite() || !particle.affine_velocity.allFinite()) {
        throw std::invalid_argument(prefix + "state must be finite");
    }
    if (!std::isfinite(particle.mass) || particle.mass <= Scalar{0}) {
        throw std::invalid_argument(prefix + "mass must be finite and positive");
    }
    if (!std::isfinite(particle.reference_volume) || particle.reference_volume <= Scalar{0}) {
        throw std::invalid_argument(prefix + "reference_volume must be finite and positive");
    }
    if (!std::isfinite(particle.plastic_volume_ratio) || particle.plastic_volume_ratio <= Scalar{0}) {
        throw std::invalid_argument(prefix + "plastic_volume_ratio must be finite and positive");
    }
    const Scalar determinant = particle.elastic_deformation.determinant();
    if (!std::isfinite(determinant) || determinant <= Scalar{0}) {
        throw std::invalid_argument(prefix + "elastic_deformation must have positive determinant");
    }
}

} // namespace nuclear_mpm
