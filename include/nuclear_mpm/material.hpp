#pragma once

#include "particle.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace nuclear_mpm {

struct LameParameters {
    Scalar mu;
    Scalar lambda;
};

[[nodiscard]] inline auto lame_parameters(Scalar young_modulus, Scalar poisson_ratio) -> LameParameters {
    if (!std::isfinite(young_modulus) || young_modulus <= Scalar{0}) {
        throw std::invalid_argument("young_modulus must be finite and positive");
    }
    if (!std::isfinite(poisson_ratio) || poisson_ratio <= Scalar{-1} || poisson_ratio >= Scalar{0.5}) {
        throw std::invalid_argument("poisson_ratio must be finite and in (-1, 0.5)");
    }
    return {
            young_modulus / (Scalar{2} * (Scalar{1} + poisson_ratio)),
            young_modulus * poisson_ratio /
                    ((Scalar{1} + poisson_ratio) * (Scalar{1} - Scalar{2} * poisson_ratio)),
    };
}

template<int Dim>
struct FixedCorotatedSolid {
    explicit FixedCorotatedSolid(Scalar young_modulus_value, Scalar poisson_ratio_value)
        : young_modulus(young_modulus_value), poisson_ratio(poisson_ratio_value) {
        (void) lame_parameters(young_modulus, poisson_ratio);
    }

    Scalar young_modulus;
    Scalar poisson_ratio;
};

template<int Dim>
struct Snow {
    explicit Snow(Scalar young_modulus_value, Scalar poisson_ratio_value, Scalar hardening_value = Scalar{10},
                  Scalar compression_limit_value = Scalar{2.5e-2}, Scalar stretch_limit_value = Scalar{4.5e-3},
                  Scalar min_plastic_volume_value = Scalar{0.6}, Scalar max_plastic_volume_value = Scalar{20})
        : young_modulus(young_modulus_value), poisson_ratio(poisson_ratio_value), hardening(hardening_value),
          compression_limit(compression_limit_value), stretch_limit(stretch_limit_value),
          min_plastic_volume(min_plastic_volume_value), max_plastic_volume(max_plastic_volume_value) {
        (void) lame_parameters(young_modulus, poisson_ratio);
        if (!std::isfinite(hardening) || hardening < Scalar{0}) {
            throw std::invalid_argument("snow hardening must be finite and non-negative");
        }
        if (!std::isfinite(compression_limit) || compression_limit < Scalar{0} || compression_limit >= Scalar{1} ||
            !std::isfinite(stretch_limit) || stretch_limit < Scalar{0}) {
            throw std::invalid_argument("snow elastic limits are invalid");
        }
        if (!std::isfinite(min_plastic_volume) || !std::isfinite(max_plastic_volume) ||
            min_plastic_volume <= Scalar{0} || max_plastic_volume < min_plastic_volume) {
            throw std::invalid_argument("snow plastic volume bounds are invalid");
        }
    }

    Scalar young_modulus;
    Scalar poisson_ratio;
    Scalar hardening;
    Scalar compression_limit;
    Scalar stretch_limit;
    Scalar min_plastic_volume;
    Scalar max_plastic_volume;
};

template<int Dim>
struct WeaklyCompressibleLiquid {
    explicit WeaklyCompressibleLiquid(Scalar bulk_modulus_value) : bulk_modulus(bulk_modulus_value) {
        if (!std::isfinite(bulk_modulus) || bulk_modulus <= Scalar{0}) {
            throw std::invalid_argument("liquid bulk_modulus must be finite and positive");
        }
    }

    Scalar bulk_modulus;
};

template<int Dim>
using Material = std::variant<FixedCorotatedSolid<Dim>, Snow<Dim>, WeaklyCompressibleLiquid<Dim>>;

template<int Dim>
class MaterialTable {
  public:
    template<typename Model>
    auto add(Model model) -> std::size_t {
        materials_.emplace_back(std::move(model));
        return materials_.size() - 1;
    }

    [[nodiscard]] auto at(std::size_t material_id) const -> const Material<Dim> & {
        if (material_id >= materials_.size()) { throw std::out_of_range("particle material_id is not registered"); }
        return materials_[material_id];
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t { return materials_.size(); }
    [[nodiscard]] auto empty() const noexcept -> bool { return materials_.empty(); }

  private:
    std::vector<Material<Dim>> materials_;
};

namespace detail {

template<int Dim>
[[nodiscard]] auto checked_deformation_determinant(const Matrix<Dim> &deformation) -> Scalar {
    if (!deformation.allFinite()) { throw std::domain_error("deformation must be finite"); }
    const Scalar determinant = deformation.determinant();
    if (!std::isfinite(determinant) || determinant <= Scalar{0}) {
        throw std::domain_error("deformation must have positive finite determinant");
    }
    return determinant;
}

template<int Dim>
[[nodiscard]] auto fixed_corotated_stress(const Matrix<Dim> &deformation, Scalar mu, Scalar lambda) -> Matrix<Dim> {
    const Scalar determinant = checked_deformation_determinant(deformation);
    const auto rotation = polar_decomposition<Dim>(deformation).rotation;
    return Scalar{2} * mu * (deformation - rotation) * deformation.transpose() +
           lambda * (determinant - Scalar{1}) * determinant * scaled_identity<Dim>(Scalar{1});
}

} // namespace detail

template<int Dim>
[[nodiscard]] auto kirchhoff_stress(const Material<Dim> &material, const Particle<Dim> &particle) -> Matrix<Dim> {
    return std::visit(
            [&](const auto &model) -> Matrix<Dim> {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (std::is_same_v<Model, FixedCorotatedSolid<Dim>>) {
                    const auto parameters = lame_parameters(model.young_modulus, model.poisson_ratio);
                    return detail::fixed_corotated_stress(particle.elastic_deformation, parameters.mu,
                                                          parameters.lambda);
                } else if constexpr (std::is_same_v<Model, Snow<Dim>>) {
                    if (!std::isfinite(particle.plastic_volume_ratio) || particle.plastic_volume_ratio <= Scalar{0}) {
                        throw std::domain_error("snow plastic_volume_ratio must be finite and positive");
                    }
                    const Scalar multiplier =
                            std::exp(model.hardening * (Scalar{1} - particle.plastic_volume_ratio));
                    if (!std::isfinite(multiplier)) { throw std::domain_error("snow hardening overflowed"); }
                    const auto parameters = lame_parameters(model.young_modulus, model.poisson_ratio);
                    return detail::fixed_corotated_stress(particle.elastic_deformation, parameters.mu * multiplier,
                                                          parameters.lambda * multiplier);
                } else {
                    const Scalar determinant =
                            detail::checked_deformation_determinant(particle.elastic_deformation);
                    return model.bulk_modulus * (determinant - Scalar{1}) * determinant *
                           scaled_identity<Dim>(Scalar{1});
                }
            },
            material);
}

template<int Dim>
[[nodiscard]] auto project_deformation(const Material<Dim> &material, const Matrix<Dim> &trial_deformation,
                                       Scalar &plastic_volume_ratio) -> Matrix<Dim> {
    const Scalar trial_determinant = detail::checked_deformation_determinant(trial_deformation);
    return std::visit(
            [&](const auto &model) -> Matrix<Dim> {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (std::is_same_v<Model, FixedCorotatedSolid<Dim>>) {
                    return trial_deformation;
                } else if constexpr (std::is_same_v<Model, Snow<Dim>>) {
                    const auto decomposition = signed_svd<Dim>(trial_deformation);
                    auto projected_values = decomposition.singular_values;
                    for (int axis = 0; axis < Dim; ++axis) {
                        projected_values(axis) =
                                std::clamp(projected_values(axis), Scalar{1} - model.compression_limit,
                                           Scalar{1} + model.stretch_limit);
                    }
                    const Matrix<Dim> projected =
                            decomposition.u * projected_values.asDiagonal() * decomposition.v.transpose();
                    const Scalar projected_determinant = detail::checked_deformation_determinant(projected);
                    plastic_volume_ratio =
                            std::clamp(plastic_volume_ratio * trial_determinant / projected_determinant,
                                       model.min_plastic_volume, model.max_plastic_volume);
                    return projected;
                } else {
                    const Scalar isotropic_stretch = std::pow(trial_determinant, Scalar{1} / Scalar{Dim});
                    if (!std::isfinite(isotropic_stretch)) {
                        throw std::domain_error("liquid deformation projection is non-finite");
                    }
                    return scaled_identity<Dim>(isotropic_stretch);
                }
            },
            material);
}

template<int Dim>
[[nodiscard]] auto characteristic_wave_speed(const Material<Dim> &material, Scalar density) -> Scalar {
    if (!std::isfinite(density) || density <= Scalar{0}) {
        throw std::invalid_argument("density must be finite and positive");
    }

    return std::visit(
            [&](const auto &model) -> Scalar {
                using Model = std::decay_t<decltype(model)>;
                if constexpr (std::is_same_v<Model, WeaklyCompressibleLiquid<Dim>>) {
                    return std::sqrt(model.bulk_modulus / density);
                } else {
                    const auto parameters = lame_parameters(model.young_modulus, model.poisson_ratio);
                    Scalar multiplier = Scalar{1};
                    if constexpr (std::is_same_v<Model, Snow<Dim>>) {
                        multiplier = std::exp(model.hardening * (Scalar{1} - model.min_plastic_volume));
                    }
                    const Scalar speed =
                            std::sqrt(multiplier * (parameters.lambda + Scalar{2} * parameters.mu) / density);
                    if (!std::isfinite(speed)) { throw std::domain_error("material wave speed is non-finite"); }
                    return speed;
                }
            },
            material);
}

} // namespace nuclear_mpm
