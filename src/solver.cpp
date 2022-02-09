#include "nclr.h"
#include <cstdint>
#include <flags.h>
#include <iostream>
#include <memory>

// The color to paint the points
constexpr int kColor = 0xED553B;

// The MPM Grid Resolution (Change at your own risk!)
constexpr int kGridResolution = 80;

// The MPM Timestep (Change at your own risk!)
constexpr nclr::real kDt = 1e-4;

auto help_msg() -> void {
    std::cout << "Usage: ./nuclear_mpm_solver [OPTIONS] COMMAND [ARGS]..." << std::endl;
    std::cout << "\tNuclearMPM headless solver" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "\t--steps\tINTEGER\t[default:1000]\tThe number of simulation steps" << std::endl;
    std::cout << "\t--cubes\tINTEGER\t[default:1]\tThe number of cubes to add" << std::endl;
    std::cout << "\t--cube-res\tINTEGER\t[default:25]\tThe resolution of each cube" << std::endl;
    std::cout << "\t--dim\tINTEGER\t[default:2]\tThe number of dimensions to run the sim in [2d or 3d only!]"
              << std::endl;
    std::cout << "\t--E\tFLOAT\t[default:1000.0]\tThe young's modulus of the shape(s)" << std::endl;
    std::cout << "\t--nu\tFLOAT\t[default:0.3]\tThe poisson's ratio of the shape(s)" << std::endl;
    std::cout << "\t--gravity\tFLOAT\t[default:-9.8]\tThe gravitational forces" << std::endl;
    std::cout
            << "\t--cube[n]-[xyz]\t\tEach cube gets its own position, this _must_ be explicitly set (0.1-0.9 for each)"
            << std::endl;
    std::cout << "\t--help\tShow this message and exit" << std::endl;
}

template<typename Sim, int dim>
auto solve_mpm(const Sim &sim, int steps) -> std::vector<nclr::Particle<dim>> {
    for (int step = 0; step < steps; ++step) {}
}

int main(int argc, char **argv) {
    const flags::args args(argc, argv);
    const auto steps = args.get<int>("steps");
    const auto cubes = args.get<int>("cubes");
    const auto cube_res = args.get<int>("cube-res");
    const auto dim = args.get<int>("dim");
    const auto E = args.get<nclr::real>("E");
    const auto nu = args.get<nclr::real>("nu");
    const auto gravity = args.get<nclr::real>("gravity");
    const auto help = args.get<int>("help", false);

    if (help || !steps && !cubes && !cube_res && !dim && !E && !nu) { help_msg(); }

    // Ew
    if (dim.value_or(2) == 2) {
        auto particles = std::vector<nclr::Particle<2>>{};
        auto sim = std::make_unique<nclr::MPMSimulation<2>>(particles, kGridResolution, kDt, E.value_or(1000.0),
                                                            nu.value_or(0.3), gravity.value_or(-9.8));
    } else {
        auto particles = std::vector<nclr::Particle<3>>{};
        auto sim = std::make_unique<nclr::MPMSimulation<3>>(particles, kGridResolution, kDt, E.value_or(1000.0),
                                                            nu.value_or(0.3), gravity.value_or(-9.8));
    }
}
