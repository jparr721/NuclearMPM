#include "snapshot_io.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuclear_mpm::io {
namespace {

constexpr int kSnapshotVersion = 1;

auto require_stream(bool condition, const std::string &message) -> void {
    if (!condition) { throw std::runtime_error("invalid NuclearMPM snapshot: " + message); }
}

auto expect_label(std::istream &stream, const char *expected) -> void {
    std::string actual;
    require_stream(static_cast<bool>(stream >> actual), std::string("missing ") + expected);
    require_stream(actual == expected, std::string("expected ") + expected + ", found " + actual);
}

template<int Dim>
auto write_vector(std::ostream &stream, const Vector<Dim> &value) -> void {
    for (int axis = 0; axis < Dim; ++axis) { stream << ' ' << value(axis); }
}

template<int Dim>
auto read_vector(std::istream &stream, Vector<Dim> &value) -> void {
    for (int axis = 0; axis < Dim; ++axis) {
        require_stream(static_cast<bool>(stream >> value(axis)), "truncated vector");
    }
}

template<int Dim>
auto write_index(std::ostream &stream, const Index<Dim> &value) -> void {
    for (int axis = 0; axis < Dim; ++axis) { stream << ' ' << value(axis); }
}

template<int Dim>
auto read_index(std::istream &stream, Index<Dim> &value) -> void {
    for (int axis = 0; axis < Dim; ++axis) {
        require_stream(static_cast<bool>(stream >> value(axis)), "truncated index");
    }
}

template<int Dim>
auto write_matrix(std::ostream &stream, const Matrix<Dim> &value) -> void {
    for (int row = 0; row < Dim; ++row) {
        for (int column = 0; column < Dim; ++column) { stream << ' ' << value(row, column); }
    }
}

template<int Dim>
auto read_matrix(std::istream &stream, Matrix<Dim> &value) -> void {
    for (int row = 0; row < Dim; ++row) {
        for (int column = 0; column < Dim; ++column) {
            require_stream(static_cast<bool>(stream >> value(row, column)), "truncated matrix");
        }
    }
}

template<int Dim>
auto write_diagnostics(std::ostream &stream, const StepDiagnostics<Dim> &diagnostics) -> void {
    stream << "substep_count " << diagnostics.substep_count << '\n';
    stream << "time_step_range " << diagnostics.minimum_time_step << ' ' << diagnostics.maximum_time_step << '\n';
    stream << "mass " << diagnostics.particle_mass << ' ' << diagnostics.grid_mass << '\n';
    stream << "linear_momentum";
    write_vector(stream, diagnostics.linear_momentum);
    stream << '\n';
    stream << "maximum_speeds " << diagnostics.maximum_particle_speed << ' ' << diagnostics.maximum_grid_speed << '\n';
    stream << "deformation_determinants " << diagnostics.minimum_deformation_determinant << ' '
           << diagnostics.maximum_deformation_determinant << '\n';
    stream << "particle_bounds_min";
    write_vector(stream, diagnostics.particle_bounds_min);
    stream << '\n';
    stream << "particle_bounds_max";
    write_vector(stream, diagnostics.particle_bounds_max);
    stream << '\n';
    stream << "outflow " << diagnostics.removed_outflow_particles << ' ' << diagnostics.removed_outflow_mass << '\n';
}

template<int Dim>
auto read_diagnostics(std::istream &stream, StepDiagnostics<Dim> &diagnostics) -> void {
    expect_label(stream, "substep_count");
    require_stream(static_cast<bool>(stream >> diagnostics.substep_count), "invalid substep_count");
    expect_label(stream, "time_step_range");
    require_stream(static_cast<bool>(stream >> diagnostics.minimum_time_step >> diagnostics.maximum_time_step),
                   "invalid time_step_range");
    expect_label(stream, "mass");
    require_stream(static_cast<bool>(stream >> diagnostics.particle_mass >> diagnostics.grid_mass), "invalid mass");
    expect_label(stream, "linear_momentum");
    read_vector(stream, diagnostics.linear_momentum);
    expect_label(stream, "maximum_speeds");
    require_stream(static_cast<bool>(stream >> diagnostics.maximum_particle_speed >> diagnostics.maximum_grid_speed),
                   "invalid maximum_speeds");
    expect_label(stream, "deformation_determinants");
    require_stream(static_cast<bool>(stream >> diagnostics.minimum_deformation_determinant >>
                                     diagnostics.maximum_deformation_determinant),
                   "invalid deformation_determinants");
    expect_label(stream, "particle_bounds_min");
    read_vector(stream, diagnostics.particle_bounds_min);
    expect_label(stream, "particle_bounds_max");
    read_vector(stream, diagnostics.particle_bounds_max);
    expect_label(stream, "outflow");
    require_stream(static_cast<bool>(stream >> diagnostics.removed_outflow_particles >>
                                     diagnostics.removed_outflow_mass),
                   "invalid outflow diagnostics");
}

} // namespace

template<int Dim>
auto write_snapshot(const std::filesystem::path &path, Scalar simulation_time,
                    const SimulationConfig<Dim> &config, const std::vector<Particle<Dim>> &particles,
                    const Grid<Dim> &grid, const StepDiagnostics<Dim> &diagnostics) -> void {
    if (!std::isfinite(simulation_time) || simulation_time < Scalar{0}) {
        throw std::invalid_argument("snapshot simulation_time must be finite and non-negative");
    }
    if (grid.cells_per_axis() != config.cells_per_axis) {
        throw std::invalid_argument("snapshot grid does not match simulation configuration");
    }
    if (path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }

    std::ofstream stream(path, std::ios::trunc);
    if (!stream) { throw std::runtime_error("failed to open snapshot for writing: " + path.string()); }
    stream << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    stream << "NUCLEAR_MPM_SNAPSHOT " << kSnapshotVersion << '\n';
    stream << "dimension " << Dim << '\n';
    stream << "simulation_time " << simulation_time << '\n';
    stream << "cells_per_axis " << config.cells_per_axis << '\n';
    stream << "particle_count " << particles.size() << '\n';
    stream << "grid_node_count " << grid.node_count() << '\n';
    write_diagnostics(stream, diagnostics);
    stream << "particles\n";
    for (const auto &particle : particles) {
        stream << "particle";
        write_vector(stream, particle.position);
        write_vector(stream, particle.velocity);
        write_matrix(stream, particle.elastic_deformation);
        write_matrix(stream, particle.affine_velocity);
        stream << ' ' << particle.plastic_volume_ratio << ' ' << particle.mass << ' ' << particle.reference_volume
               << ' ' << particle.material_id << ' ' << particle.user_tag << '\n';
    }
    stream << "grid_nodes\n";
    grid.for_each_node([&](const Index<Dim> &coordinate, const GridNode<Dim> &node) {
        stream << "node";
        write_index(stream, coordinate);
        stream << ' ' << node.mass;
        write_vector(stream, node.momentum);
        write_vector(stream, node.velocity);
        stream << '\n';
    });
    stream << "end\n";
    if (!stream) { throw std::runtime_error("failed while writing snapshot: " + path.string()); }
}

template<int Dim>
[[nodiscard]] auto read_snapshot(const std::filesystem::path &path) -> SnapshotData<Dim> {
    std::ifstream stream(path);
    if (!stream) { throw std::runtime_error("failed to open snapshot for reading: " + path.string()); }

    SnapshotData<Dim> snapshot;
    expect_label(stream, "NUCLEAR_MPM_SNAPSHOT");
    require_stream(static_cast<bool>(stream >> snapshot.version), "missing version");
    require_stream(snapshot.version == kSnapshotVersion, "unsupported version");

    expect_label(stream, "dimension");
    int dimension = 0;
    require_stream(static_cast<bool>(stream >> dimension), "missing dimension");
    require_stream(dimension == Dim, "dimension does not match requested snapshot type");
    expect_label(stream, "simulation_time");
    require_stream(static_cast<bool>(stream >> snapshot.simulation_time), "invalid simulation_time");
    expect_label(stream, "cells_per_axis");
    require_stream(static_cast<bool>(stream >> snapshot.cells_per_axis), "invalid cells_per_axis");

    std::size_t particle_count = 0;
    std::size_t grid_node_count = 0;
    expect_label(stream, "particle_count");
    require_stream(static_cast<bool>(stream >> particle_count), "invalid particle_count");
    expect_label(stream, "grid_node_count");
    require_stream(static_cast<bool>(stream >> grid_node_count), "invalid grid_node_count");
    read_diagnostics(stream, snapshot.diagnostics);

    expect_label(stream, "particles");
    snapshot.particles.reserve(particle_count);
    for (std::size_t index = 0; index < particle_count; ++index) {
        expect_label(stream, "particle");
        Particle<Dim> particle;
        read_vector(stream, particle.position);
        read_vector(stream, particle.velocity);
        read_matrix(stream, particle.elastic_deformation);
        read_matrix(stream, particle.affine_velocity);
        require_stream(static_cast<bool>(stream >> particle.plastic_volume_ratio >> particle.mass >>
                                         particle.reference_volume >> particle.material_id >> particle.user_tag),
                       "truncated particle");
        validate_particle(particle, index);
        snapshot.particles.push_back(particle);
    }

    Grid<Dim> expected_grid(snapshot.cells_per_axis);
    require_stream(grid_node_count == expected_grid.node_count(), "grid_node_count does not match topology");
    std::vector<bool> visited(grid_node_count, false);
    expect_label(stream, "grid_nodes");
    snapshot.grid_nodes.reserve(grid_node_count);
    for (std::size_t index = 0; index < grid_node_count; ++index) {
        expect_label(stream, "node");
        GridNodeSnapshot<Dim> node;
        read_index(stream, node.coordinate);
        require_stream(static_cast<bool>(stream >> node.mass), "invalid node mass");
        read_vector(stream, node.momentum);
        read_vector(stream, node.velocity);
        const auto flat = expected_grid.flatten(node.coordinate);
        require_stream(!visited[flat], "duplicate grid coordinate");
        visited[flat] = true;
        snapshot.grid_nodes.push_back(node);
    }
    expect_label(stream, "end");
    return snapshot;
}

template auto write_snapshot<2>(const std::filesystem::path &, Scalar, const SimulationConfig<2> &,
                                const std::vector<Particle<2>> &, const Grid<2> &,
                                const StepDiagnostics<2> &) -> void;
template auto write_snapshot<3>(const std::filesystem::path &, Scalar, const SimulationConfig<3> &,
                                const std::vector<Particle<3>> &, const Grid<3> &,
                                const StepDiagnostics<3> &) -> void;
template auto read_snapshot<2>(const std::filesystem::path &) -> SnapshotData<2>;
template auto read_snapshot<3>(const std::filesystem::path &) -> SnapshotData<3>;

} // namespace nuclear_mpm::io
