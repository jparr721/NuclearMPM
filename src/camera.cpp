#include "camera.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nuclear_mpm::viz {
namespace {

constexpr float kPi = 3.14159265358979323846F;

inline auto require_finite(float value, const char *name) -> void {
    if (!std::isfinite(value)) { throw std::invalid_argument(std::string(name) + " must be finite"); }
}

inline auto validate_bounds(const DomainBounds &bounds) -> Eigen::Vector3f {
    if (!bounds.minimum.allFinite() || !bounds.maximum.allFinite()) {
        throw std::invalid_argument("camera domain bounds must be finite");
    }
    const Eigen::Vector3f extent = bounds.maximum - bounds.minimum;
    if ((extent.array() <= 0.0F).any()) {
        throw std::invalid_argument("camera domain maximum must exceed minimum");
    }
    return extent;
}

inline auto look_at(const Eigen::Vector3f &eye, const Eigen::Vector3f &target,
                    const Eigen::Vector3f &world_up = Eigen::Vector3f::UnitY()) -> Eigen::Matrix4f {
    const Eigen::Vector3f forward = (target - eye).normalized();
    const Eigen::Vector3f right = forward.cross(world_up).normalized();
    const Eigen::Vector3f up = right.cross(forward);

    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result.block<1, 3>(0, 0) = right.transpose();
    result.block<1, 3>(1, 0) = up.transpose();
    result.block<1, 3>(2, 0) = -forward.transpose();
    result(0, 3) = -right.dot(eye);
    result(1, 3) = -up.dot(eye);
    result(2, 3) = forward.dot(eye);
    return result;
}

inline auto perspective(float vertical_fov_radians, float aspect_ratio, float near_plane,
                        float far_plane) -> Eigen::Matrix4f {
    require_finite(aspect_ratio, "camera aspect_ratio");
    if (aspect_ratio <= 0.0F || near_plane <= 0.0F || far_plane <= near_plane) {
        throw std::invalid_argument("invalid perspective projection bounds");
    }
    const float scale = 1.0F / std::tan(vertical_fov_radians * 0.5F);
    Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
    result(0, 0) = scale / aspect_ratio;
    result(1, 1) = scale;
    result(2, 2) = (far_plane + near_plane) / (near_plane - far_plane);
    result(2, 3) = (2.0F * far_plane * near_plane) / (near_plane - far_plane);
    result(3, 2) = -1.0F;
    return result;
}

inline auto orthographic(float half_height, float aspect_ratio) -> Eigen::Matrix4f {
    require_finite(aspect_ratio, "camera aspect_ratio");
    if (aspect_ratio <= 0.0F || half_height <= 0.0F) {
        throw std::invalid_argument("invalid orthographic projection bounds");
    }
    const float half_width = half_height * aspect_ratio;
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result(0, 0) = 1.0F / half_width;
    result(1, 1) = 1.0F / half_height;
    result(2, 2) = -0.1F;
    return result;
}

} // namespace

auto operator==(const CameraInput &left, const CameraInput &right) noexcept -> bool {
    return left.orbit_x == right.orbit_x && left.orbit_y == right.orbit_y && left.pan_x == right.pan_x &&
           left.pan_y == right.pan_y && left.dolly == right.dolly;
}

auto filter_camera_input(const CameraInput &input, bool ui_wants_mouse) noexcept -> CameraInput {
    return ui_wants_mouse ? CameraInput{} : input;
}

OrthographicCamera::OrthographicCamera() { reset(DomainBounds{}); }

auto OrthographicCamera::reset(const DomainBounds &bounds) -> void {
    const auto extent = validate_bounds(bounds);
    target_ = (bounds.minimum + bounds.maximum) * 0.5F;
    half_height_ = std::max(extent.x(), extent.y()) * 0.6F;
}

auto OrthographicCamera::pan(const Eigen::Vector2f &world_delta) -> void {
    if (!world_delta.allFinite()) { throw std::invalid_argument("orthographic pan must be finite"); }
    target_.x() += world_delta.x();
    target_.y() += world_delta.y();
}

auto OrthographicCamera::zoom(float amount) -> void {
    require_finite(amount, "orthographic zoom");
    half_height_ = std::clamp(half_height_ * std::exp(-amount), 0.01F, 100.0F);
}

auto OrthographicCamera::target() const noexcept -> const Eigen::Vector3f & { return target_; }
auto OrthographicCamera::half_height() const noexcept -> float { return half_height_; }

auto OrthographicCamera::view_matrix() const -> Eigen::Matrix4f {
    return look_at(target_ + Eigen::Vector3f{0.0F, 0.0F, 2.0F}, target_);
}

auto OrthographicCamera::projection_matrix(float aspect_ratio) const -> Eigen::Matrix4f {
    return orthographic(half_height_, aspect_ratio);
}

OrbitCamera::OrbitCamera() { reset(DomainBounds{}); }

auto OrbitCamera::reset(const DomainBounds &bounds) -> void {
    const auto extent = validate_bounds(bounds);
    target_ = (bounds.minimum + bounds.maximum) * 0.5F;
    domain_radius_ = std::max(extent.norm() * 0.5F, 0.1F);
    distance_ = std::clamp(domain_radius_ * 3.0F, 0.05F, 100.0F);
    yaw_ = kPi * 0.25F;
    pitch_ = kPi * 0.15F;
}

auto OrbitCamera::orbit(const Eigen::Vector2f &radians) -> void {
    if (!radians.allFinite()) { throw std::invalid_argument("orbit delta must be finite"); }
    yaw_ += radians.x();
    constexpr float pitch_limit = kPi * 0.494F;
    pitch_ = std::clamp(pitch_ + radians.y(), -pitch_limit, pitch_limit);
}

auto OrbitCamera::pan(const Eigen::Vector2f &world_delta) -> void {
    if (!world_delta.allFinite()) { throw std::invalid_argument("orbit pan must be finite"); }
    const Eigen::Vector3f forward = (target_ - eye()).normalized();
    const Eigen::Vector3f right = forward.cross(Eigen::Vector3f::UnitY()).normalized();
    const Eigen::Vector3f up = right.cross(forward);
    target_ += world_delta.x() * right + world_delta.y() * up;
}

auto OrbitCamera::dolly(float amount) -> void {
    require_finite(amount, "orbit dolly");
    distance_ = std::clamp(distance_ * std::exp(-amount), 0.05F, 100.0F);
}

auto OrbitCamera::eye() const -> Eigen::Vector3f {
    const float horizontal = distance_ * std::cos(pitch_);
    return target_ + Eigen::Vector3f{
                             horizontal * std::cos(yaw_),
                             distance_ * std::sin(pitch_),
                             horizontal * std::sin(yaw_),
                     };
}

auto OrbitCamera::target() const noexcept -> const Eigen::Vector3f & { return target_; }
auto OrbitCamera::distance() const noexcept -> float { return distance_; }
auto OrbitCamera::view_matrix() const -> Eigen::Matrix4f { return look_at(eye(), target_); }

auto OrbitCamera::projection_matrix(float aspect_ratio) const -> Eigen::Matrix4f {
    const float near_plane = std::max(0.001F, distance_ - domain_radius_ * 2.0F);
    const float far_plane = std::max(near_plane + 1.0F, distance_ + domain_radius_ * 4.0F);
    return perspective(kPi * 0.25F, aspect_ratio, near_plane, far_plane);
}

} // namespace nuclear_mpm::viz
