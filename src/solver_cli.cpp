#include "solver_cli.h"

#include "snapshot_io.h"

#include <flags.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuclear_mpm::cli {
namespace {

template<typename T, typename Parser>
[[nodiscard]] auto parse_list(const std::string &text, Parser &&parser, const char *option_name) -> std::vector<T> {
    std::vector<T> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) { throw std::invalid_argument(std::string(option_name) + " contains an empty component"); }
        values.push_back(parser(token));
    }
    if (values.empty()) { throw std::invalid_argument(std::string(option_name) + " must not be empty"); }
    return values;
}

[[nodiscard]] auto parse_scalar(const std::string &text) -> Scalar {
    std::size_t consumed = 0;
    const Scalar value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite scalar: " + text);
    }
    return value;
}

[[nodiscard]] auto parse_integer(const std::string &text) -> int {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) { throw std::invalid_argument("invalid integer: " + text); }
    return value;
}

template<typename T>
auto require_dimension(const std::vector<T> &values, int dimension, const char *option_name) -> void {
    if (values.size() != static_cast<std::size_t>(dimension)) {
        throw std::invalid_argument(std::string(option_name) + " requires exactly " + std::to_string(dimension) +
                                    " comma-separated components");
    }
}

template<int Dim>
[[nodiscard]] auto fixed_vector(const std::vector<Scalar> &values) -> Vector<Dim> {
    require_dimension(values, Dim, "vector");
    Vector<Dim> result;
    for (int axis = 0; axis < Dim; ++axis) { result(axis) = values[static_cast<std::size_t>(axis)]; }
    return result;
}

template<int Dim>
[[nodiscard]] auto fixed_index(const std::vector<int> &values) -> Index<Dim> {
    require_dimension(values, Dim, "index");
    Index<Dim> result;
    for (int axis = 0; axis < Dim; ++axis) { result(axis) = values[static_cast<std::size_t>(axis)]; }
    return result;
}

template<int Dim>
auto run_dimension(const SolverOptions &options) -> int {
    SimulationConfig<Dim> config;
    config.cells_per_axis = options.cells_per_axis;
    config.max_time_step = options.max_time_step;
    config.cfl_factor = options.cfl_factor;
    switch (options.material) {
        case MaterialKind::Solid:
            config.materials.add(FixedCorotatedSolid<Dim>(Scalar{1000}, Scalar{0.2}));
            break;
        case MaterialKind::Snow:
            config.materials.add(Snow<Dim>(Scalar{1000}, Scalar{0.2}));
            break;
        case MaterialKind::Liquid:
            config.materials.add(WeaklyCompressibleLiquid<Dim>(Scalar{1000}));
            break;
    }

    RuntimeControls<Dim> controls;
    controls.gravity = fixed_vector<Dim>(options.gravity);
    auto particles = sample_box_lattice<Dim>(fixed_index<Dim>(options.points_per_axis),
                                              fixed_vector<Dim>(options.box_min), fixed_vector<Dim>(options.box_max),
                                              options.density, 0, 0xED553BU);
    Simulation<Dim> simulation(std::move(config), controls, std::move(particles));
    const auto diagnostics = simulation.advance(options.duration);

    if (options.output) {
        io::write_snapshot(*options.output, simulation.simulation_time(), simulation.config(), simulation.particles(),
                           simulation.grid(), diagnostics);
    }

    std::cout << "Simulation complete: dimension=" << Dim << " particles=" << simulation.particles().size()
              << " substeps=" << diagnostics.substep_count << " time=" << simulation.simulation_time() << '\n';
    return EXIT_SUCCESS;
}

} // namespace

[[nodiscard]] auto parse_solver_options(int argc, char **argv) -> SolverOptions {
    const flags::args arguments(argc, argv);
    SolverOptions options;
    options.help = arguments.get<bool>("help", false);

    const int dimension = arguments.get<int>("dimension").value_or(2);
    if (dimension == 2) {
        options.dimension = RuntimeDimension::Two;
        options.box_min = {Scalar{0.4}, Scalar{0.4}};
        options.box_max = {Scalar{0.6}, Scalar{0.6}};
        options.points_per_axis = {16, 16};
        options.gravity = {Scalar{0}, Scalar{-9.81}};
    } else if (dimension == 3) {
        options.dimension = RuntimeDimension::Three;
        options.box_min = {Scalar{0.4}, Scalar{0.4}, Scalar{0.4}};
        options.box_max = {Scalar{0.6}, Scalar{0.6}, Scalar{0.6}};
        options.points_per_axis = {8, 8, 8};
        options.gravity = {Scalar{0}, Scalar{-9.81}, Scalar{0}};
    } else {
        throw std::invalid_argument("--dimension must be 2 or 3");
    }

    const auto material = arguments.get<std::string>("material").value_or("snow");
    if (material == "solid") {
        options.material = MaterialKind::Solid;
    } else if (material == "snow") {
        options.material = MaterialKind::Snow;
    } else if (material == "liquid") {
        options.material = MaterialKind::Liquid;
    } else {
        throw std::invalid_argument("--material must be solid, snow, or liquid");
    }

    const int dimensions = static_cast<int>(options.dimension);
    if (const auto value = arguments.get<std::string>("box-min")) {
        options.box_min = parse_list<Scalar>(*value, parse_scalar, "--box-min");
    }
    if (const auto value = arguments.get<std::string>("box-max")) {
        options.box_max = parse_list<Scalar>(*value, parse_scalar, "--box-max");
    }
    if (const auto value = arguments.get<std::string>("points-per-axis")) {
        options.points_per_axis = parse_list<int>(*value, parse_integer, "--points-per-axis");
    }
    if (const auto value = arguments.get<std::string>("gravity")) {
        options.gravity = parse_list<Scalar>(*value, parse_scalar, "--gravity");
    }
    require_dimension(options.box_min, dimensions, "--box-min");
    require_dimension(options.box_max, dimensions, "--box-max");
    require_dimension(options.points_per_axis, dimensions, "--points-per-axis");
    require_dimension(options.gravity, dimensions, "--gravity");

    options.cells_per_axis = arguments.get<int>("cells").value_or(options.cells_per_axis);
    options.duration = arguments.get<Scalar>("duration").value_or(options.duration);
    options.max_time_step = arguments.get<Scalar>("max-dt").value_or(options.max_time_step);
    options.cfl_factor = arguments.get<Scalar>("cfl").value_or(options.cfl_factor);
    options.density = arguments.get<Scalar>("density").value_or(options.density);
    if (const auto output = arguments.get<std::string>("output")) { options.output = *output; }

    if (options.cells_per_axis <= 0 || !std::isfinite(options.duration) || options.duration <= Scalar{0} ||
        !std::isfinite(options.max_time_step) || options.max_time_step <= Scalar{0} ||
        !std::isfinite(options.cfl_factor) || options.cfl_factor <= Scalar{0} || options.cfl_factor > Scalar{1} ||
        !std::isfinite(options.density) || options.density <= Scalar{0}) {
        throw std::invalid_argument("solver numeric options must be finite, positive, and physically valid");
    }
    for (const int count : options.points_per_axis) {
        if (count <= 0) { throw std::invalid_argument("--points-per-axis components must be positive"); }
    }
    return options;
}

[[nodiscard]] auto help_message() -> std::string {
    return R"(NuclearMPM solver
Usage: nuclear_mpm_solver [options]
  --dimension 2|3
  --material solid|snow|liquid
  --box-min comma-separated coordinates
  --box-max comma-separated coordinates
  --points-per-axis comma-separated counts
  --gravity comma-separated acceleration
  --cells positive integer
  --duration positive seconds
  --max-dt positive seconds
  --cfl value in (0,1]
  --density positive value
  --output snapshot path
  --help
)";
}

auto run_solver(const SolverOptions &options) -> int {
    if (options.help) {
        std::cout << help_message();
        return EXIT_SUCCESS;
    }
    switch (options.dimension) {
        case RuntimeDimension::Two:
            return run_dimension<2>(options);
        case RuntimeDimension::Three:
            return run_dimension<3>(options);
    }
    throw std::invalid_argument("unsupported runtime dimension");
}

} // namespace nuclear_mpm::cli
