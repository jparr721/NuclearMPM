#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nclr::viz {

struct Point {
    float x;
    float y;
    std::uint32_t color;
};

class Visualizer {
  public:
    Visualizer(std::string title, int width, int height);
    ~Visualizer();

    Visualizer(const Visualizer &) = delete;
    auto operator=(const Visualizer &) -> Visualizer & = delete;

    [[nodiscard]] auto should_close() const -> bool;
    auto draw(const std::vector<Point> &points) -> void;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nclr::viz
