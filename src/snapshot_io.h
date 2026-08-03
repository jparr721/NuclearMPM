#pragma once

#include <nuclear_mpm/nuclear_mpm.hpp>

#include <filesystem>
#include <vector>

namespace nuclear_mpm::io {

template<int Dim>
struct GridNodeSnapshot {
    Index<Dim> coordinate = Index<Dim>::Zero();
    Scalar mass = Scalar{0};
    Vector<Dim> momentum = Vector<Dim>::Zero();
    Vector<Dim> velocity = Vector<Dim>::Zero();
};

template<int Dim>
struct SnapshotData {
    int version = 0;
    Scalar simulation_time = Scalar{0};
    int cells_per_axis = 0;
    std::vector<Particle<Dim>> particles;
    std::vector<GridNodeSnapshot<Dim>> grid_nodes;
    StepDiagnostics<Dim> diagnostics;
};

template<int Dim>
auto write_snapshot(const std::filesystem::path &path, Scalar simulation_time,
                    const SimulationConfig<Dim> &config, const std::vector<Particle<Dim>> &particles,
                    const Grid<Dim> &grid, const StepDiagnostics<Dim> &diagnostics) -> void;

template<int Dim>
[[nodiscard]] auto read_snapshot(const std::filesystem::path &path) -> SnapshotData<Dim>;

extern template auto write_snapshot<2>(const std::filesystem::path &, Scalar, const SimulationConfig<2> &,
                                       const std::vector<Particle<2>> &, const Grid<2> &,
                                       const StepDiagnostics<2> &) -> void;
extern template auto write_snapshot<3>(const std::filesystem::path &, Scalar, const SimulationConfig<3> &,
                                       const std::vector<Particle<3>> &, const Grid<3> &,
                                       const StepDiagnostics<3> &) -> void;
extern template auto read_snapshot<2>(const std::filesystem::path &) -> SnapshotData<2>;
extern template auto read_snapshot<3>(const std::filesystem::path &) -> SnapshotData<3>;

} // namespace nuclear_mpm::io
