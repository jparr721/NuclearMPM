#pragma once

#include <Eigen/Core>

namespace nuclear_mpm::viz {

struct DomainBounds {
    Eigen::Vector3f minimum = Eigen::Vector3f::Zero();
    Eigen::Vector3f maximum = Eigen::Vector3f::Ones();
};

struct CameraInput {
    float orbit_x = 0.0F;
    float orbit_y = 0.0F;
    float pan_x = 0.0F;
    float pan_y = 0.0F;
    float dolly = 0.0F;
};

[[nodiscard]] auto operator==(const CameraInput &left, const CameraInput &right) noexcept -> bool;
[[nodiscard]] auto filter_camera_input(const CameraInput &input, bool ui_wants_mouse) noexcept -> CameraInput;
[[nodiscard]] auto map_pointer_camera_input(const Eigen::Vector2f &cursor_delta, bool orbit_active,
                                            bool pan_active, float scroll_delta) noexcept -> CameraInput;

class OrthographicCamera {
  public:
    OrthographicCamera();

    auto reset(const DomainBounds &bounds) -> void;
    auto pan(const Eigen::Vector2f &world_delta) -> void;
    auto zoom(float amount) -> void;

    [[nodiscard]] auto target() const noexcept -> const Eigen::Vector3f &;
    [[nodiscard]] auto half_height() const noexcept -> float;
    [[nodiscard]] auto view_matrix() const -> Eigen::Matrix4f;
    [[nodiscard]] auto projection_matrix(float aspect_ratio) const -> Eigen::Matrix4f;

  private:
    Eigen::Vector3f target_ = Eigen::Vector3f::Constant(0.5F);
    float half_height_ = 0.6F;
};

class OrbitCamera {
  public:
    OrbitCamera();

    auto reset(const DomainBounds &bounds) -> void;
    auto orbit(const Eigen::Vector2f &radians) -> void;
    auto pan(const Eigen::Vector2f &world_delta) -> void;
    auto dolly(float amount) -> void;

    [[nodiscard]] auto eye() const -> Eigen::Vector3f;
    [[nodiscard]] auto target() const noexcept -> const Eigen::Vector3f &;
    [[nodiscard]] auto distance() const noexcept -> float;
    [[nodiscard]] auto view_matrix() const -> Eigen::Matrix4f;
    [[nodiscard]] auto projection_matrix(float aspect_ratio) const -> Eigen::Matrix4f;

  private:
    Eigen::Vector3f target_ = Eigen::Vector3f::Constant(0.5F);
    float yaw_ = 0.785398163F;
    float pitch_ = 0.4F;
    float distance_ = 2.5F;
    float domain_radius_ = 1.0F;
};

} // namespace nuclear_mpm::viz
