#pragma once

#include <nuclear_mpm/nuclear_mpm.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nuclear_mpm::cli {

enum class MaterialKind {
    Solid,
    Snow,
    Liquid,
};

struct SolverOptions {
    bool help = false;
    RuntimeDimension dimension = RuntimeDimension::Two;
    MaterialKind material = MaterialKind::Snow;
    std::vector<Scalar> box_min;
    std::vector<Scalar> box_max;
    std::vector<int> points_per_axis;
    std::vector<Scalar> gravity;
    int cells_per_axis = 64;
    Scalar duration = Scalar{1.0e-2};
    Scalar max_time_step = Scalar{1.0e-4};
    Scalar cfl_factor = Scalar{0.4};
    Scalar density = Scalar{1};
    std::optional<std::filesystem::path> output;
};

[[nodiscard]] auto parse_solver_options(int argc, char **argv) -> SolverOptions;
[[nodiscard]] auto help_message() -> std::string;
auto run_solver(const SolverOptions &options) -> int;

} // namespace nuclear_mpm::cli
