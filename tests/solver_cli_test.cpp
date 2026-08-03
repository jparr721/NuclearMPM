#include <catch2/catch_test_macros.hpp>
#include "solver_cli.h"

#include <string>
#include <vector>

namespace {

auto parse(std::vector<std::string> arguments) -> nuclear_mpm::cli::SolverOptions {
    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (auto &argument : arguments) { argv.push_back(argument.data()); }
    return nuclear_mpm::cli::parse_solver_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST_CASE("CLI parses explicit 3D geometry", "[cli]") {
    const auto options = parse({
            "solver",
            "--dimension", "3",
            "--material", "snow",
            "--box-min", "0.2,0.3,0.4",
            "--box-max", "0.6,0.7,0.8",
            "--points-per-axis", "2,3,4",
            "--gravity", "0,-9.81,0",
            "--cells", "16",
            "--duration", "0.001",
            "--max-dt", "0.0001",
    });

    REQUIRE(options.dimension == nuclear_mpm::RuntimeDimension::Three);
    REQUIRE(options.material == nuclear_mpm::cli::MaterialKind::Snow);
    REQUIRE(options.box_min == std::vector<nuclear_mpm::Scalar>{0.2F, 0.3F, 0.4F});
    REQUIRE(options.box_max == std::vector<nuclear_mpm::Scalar>{0.6F, 0.7F, 0.8F});
    REQUIRE(options.points_per_axis == std::vector<int>{2, 3, 4});
    REQUIRE(options.gravity == std::vector<nuclear_mpm::Scalar>{0.0F, -9.81F, 0.0F});
    REQUIRE(options.cells_per_axis == 16);
}

TEST_CASE("CLI rejects unsupported dimensions and materials", "[cli]") {
    REQUIRE_THROWS_AS(parse({"solver", "--dimension", "4"}), std::invalid_argument);
    REQUIRE_THROWS_AS(parse({"solver", "--material", "mud"}), std::invalid_argument);
}

TEST_CASE("CLI rejects vectors with the wrong dimensionality", "[cli]") {
    REQUIRE_THROWS_AS(
            parse({"solver", "--dimension", "3", "--box-min", "0.2,0.3", "--box-max", "0.6,0.7,0.8"}),
            std::invalid_argument);
    REQUIRE_THROWS_AS(parse({"solver", "--dimension", "2", "--points-per-axis", "2,2,2"}),
                      std::invalid_argument);
}

TEST_CASE("CLI exposes help without constructing a simulation", "[cli]") {
    const auto options = parse({"solver", "--help"});
    REQUIRE(options.help);
}
