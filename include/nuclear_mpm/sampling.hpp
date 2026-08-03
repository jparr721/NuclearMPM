#pragma once

#include "particle.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nuclear_mpm {

template<int Dim>
[[nodiscard]] auto sample_box_lattice(const Index<Dim> &points_per_axis, const Vector<Dim> &min_corner,
                                      const Vector<Dim> &max_corner, Scalar density, std::size_t material_id,
                                      std::uint32_t user_tag) -> std::vector<Particle<Dim>> {
    static_assert(is_supported_dimension<Dim>, "NuclearMPM sampling supports only 2D and 3D");
    if (!min_corner.allFinite() || !max_corner.allFinite()) {
        throw std::invalid_argument("sample box corners must be finite");
    }
    if (!std::isfinite(density) || density <= Scalar{0}) {
        throw std::invalid_argument("sample density must be finite and positive");
    }

    std::size_t particle_count = 1;
    Scalar box_measure = Scalar{1};
    Vector<Dim> spacing;
    for (int axis = 0; axis < Dim; ++axis) {
        if (points_per_axis(axis) <= 0) {
            throw std::invalid_argument("sample points_per_axis must be positive");
        }
        if (!(max_corner(axis) > min_corner(axis))) {
            throw std::invalid_argument("sample max_corner must exceed min_corner on every axis");
        }

        const auto axis_count = static_cast<std::size_t>(points_per_axis(axis));
        if (particle_count > std::numeric_limits<std::size_t>::max() / axis_count) {
            throw std::overflow_error("sample particle count exceeds addressable storage");
        }
        particle_count *= axis_count;
        const Scalar extent = max_corner(axis) - min_corner(axis);
        box_measure *= extent;
        spacing(axis) = extent / static_cast<Scalar>(points_per_axis(axis));
    }

    const Scalar reference_volume = box_measure / static_cast<Scalar>(particle_count);
    const Scalar particle_mass = density * reference_volume;
    if (!std::isfinite(reference_volume) || !std::isfinite(particle_mass) || reference_volume <= Scalar{0} ||
        particle_mass <= Scalar{0}) {
        throw std::invalid_argument("sample mass and reference volume must be finite and positive");
    }

    std::vector<Particle<Dim>> particles;
    particles.reserve(particle_count);
    for (std::size_t ordinal = 0; ordinal < particle_count; ++ordinal) {
        std::size_t remainder = ordinal;
        Index<Dim> lattice_index;
        for (int axis = Dim - 1; axis >= 0; --axis) {
            const auto axis_count = static_cast<std::size_t>(points_per_axis(axis));
            lattice_index(axis) = static_cast<int>(remainder % axis_count);
            remainder /= axis_count;
        }

        Particle<Dim> particle;
        for (int axis = 0; axis < Dim; ++axis) {
            particle.position(axis) =
                    min_corner(axis) + (static_cast<Scalar>(lattice_index(axis)) + Scalar{0.5}) * spacing(axis);
        }
        particle.mass = particle_mass;
        particle.reference_volume = reference_volume;
        particle.material_id = material_id;
        particle.user_tag = user_tag;
        particles.push_back(particle);
    }
    return particles;
}

} // namespace nuclear_mpm
