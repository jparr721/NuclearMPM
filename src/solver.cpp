#include "nclr.h"
#include <cstdint>
#include <filesystem>
#include <flags.h>
#include <fstream>
#include <iostream>
#include <memory>
#ifdef NCLR_SOLVER_VIZ
#include "taichi.h"
#endif

namespace fs = std::filesystem;

constexpr int kWindowSize = 800;

// The color to paint the points
constexpr int kColor = 0xED553B;

// The MPM Grid Resolution (Change at your own risk!)
constexpr int kGridResolution = 64;

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
    std::cout << "\t--material-model\t[jelly, snow, liquid]\t[default:jelly]\tThe material model to use" << std::endl;
    std::cout
            << "\t--cube[n]-[xyz]\t\tEach cube gets its own position, this _must_ be explicitly set (0.1-0.9 for each)"
            << std::endl;
    std::cout << "\t--dump\tDump particle state at each timestep (impacts perforamnce)" << std::endl;
    std::cout << "\t--help\tShow this message and exit" << std::endl;
}

template<int dim, typename Sim>
auto solve_mpm(const Sim &sim, int steps, bool dump, std::vector<std::vector<nclr::Particle<dim>>> &states,
               std::vector<std::vector<nclr::Cell<dim>>> &cells) -> void {
    std::cout << "Running simulation" << std::endl;
    for (int step = 0; step < steps; ++step) {
        if (dump) {
            states.push_back(sim->particles());
            if (step > 0) {
                cells.push_back(sim->grid());
            } else {
                cells.push_back(
                        std::vector<nclr::Cell<dim>>((kGridResolution + 1) * (kGridResolution + 1), nclr::Cell<dim>()));
            }
        }
        sim->advance();
    }
    std::cout << "Simulation done" << std::endl;
}

auto exe_path() -> fs::path { return fs::canonical("."); }

template<typename T>
auto save_value(const T &value, const std::string &filename) -> void {
    const fs::path full_path = exe_path() / fs::path("tmp") / fs::path(filename);
    if (!fs::exists(full_path)) { fs::create_directories(fs::path(full_path.parent_path())); }
    std::ofstream ofs(full_path, std::fstream::in | std::fstream::out | std::fstream::app);
    ofs << value << std::endl;
    ofs.close();
}

template<int dim, typename Sim>
auto unload_particles(const std::string material_model, const Sim &sim,
                      const std::vector<std::vector<nclr::Particle<dim>>> &particles) -> void {
    std::cout << "Saving results" << std::endl;
    const std::string timestep_filename = "timestep.txt";
    const std::string x_filename = "x.txt";
    const std::string v_filename = "v.txt";
    const std::string F_filename = "F.txt";
    const std::string C_filename = "C.txt";
    const std::string Jp_filename = "Jp.txt";
    const std::string lame_filename = "lame.txt";

    int step = 0;
    for (const auto &p_list : particles) {
        const std::string prefix = std::to_string(step) + "_";
        for (const auto &p : p_list) {
            // Load the timestep
            const auto timestep = step > 0 ? kDt * step : kDt;
            save_value(timestep, prefix + timestep_filename);
            save_value(p.x, prefix + x_filename);
            save_value(p.v, prefix + v_filename);
            save_value(p.F, prefix + F_filename);
            save_value(p.C, prefix + C_filename);
            save_value(p.Jp, prefix + Jp_filename);

            const auto e = material_model == "snow"    ? sim->kSnowHardening
                           : material_model == "jelly" ? sim->kJellyHardening
                                                       : sim->kLiquidHardening;
            const auto mu = sim->mu_0 * e;
            const auto lambda = sim->lambda_0 * e;
            save_value(nclr::Vector<nclr::real, 2>(mu, lambda).transpose(), prefix + lame_filename);
        }
        ++step;
    }
    std::cout << "Done saving" << std::endl;
}

template<int dim, typename Sim>
auto unload_cells(const Sim &sim, const std::vector<std::vector<nclr::Cell<dim>>> &cells) -> void {
    const std::string mass_filename = "mass.txt";
    const std::string velocity_filename = "velocity.txt";

    const auto to_1d = [](int x, int y, int max) -> float { return (x * max) + y; };
    std::cout << "Saving grid states" << std::endl;
    int step = 0;
    for (const auto &grid_state : cells) {
        const std::string prefix = std::to_string(step) + "_";
        for (int ii = 0; ii <= kGridResolution; ++ii) {
            for (int jj = 0; jj <= kGridResolution; ++jj) {
                save_value(grid_state.at(to_1d(ii, jj, kGridResolution)).mass, prefix + mass_filename);
                save_value(grid_state.at(to_1d(ii, jj, kGridResolution)).velocity, prefix + velocity_filename);
            }
        }
        ++step;
    }
    std::cout << "Done saving" << std::endl;
}

template<int dim>
auto generate_cubes(const std::optional<int> &cubes, const std::optional<int> &cube_res, const flags::args &args)
        -> std::vector<nclr::Particle<dim>> {
    auto particles = std::vector<nclr::Particle<dim>>{};
    for (int cc = 0; cc < cubes.value_or(1); ++cc) {
        const auto cube_n_x = args.get<nclr::real>("cube" + std::to_string(cc) + "-x");
        const auto cube_n_y = args.get<nclr::real>("cube" + std::to_string(cc) + "-y");
        if (!cube_n_x || !cube_n_y) {
            std::cerr << "Cube: " << cc << " is missing coordinates" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto cube_particles = nclr::cube<dim>(cube_res.value_or(25), cube_n_x.value(), cube_n_y.value());
        for (const auto &pos : cube_particles) { particles.emplace_back(nclr::Particle<dim>(pos, kColor)); }
    }

    return particles;
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
    const auto material_model = args.get<std::string>("material-model");
    const auto dump = args.get<bool>("dump", false);
    const auto help = args.get<bool>("help", false);

    if (material_model && material_model.value() != "jelly" && material_model.value() != "snow" &&
        material_model.value() != "liquid") {
        std::cerr << "Invalid Option: " << material_model.value() << std::endl;
        help_msg();
        return EXIT_FAILURE;
    }

    if (help || !steps && !cubes && !cube_res && !dim && !E && !nu && !gravity && !material_model) { help_msg(); }

    auto model = nclr::MaterialModel::kJelly;
    if (material_model == "snow") {
        model = nclr::MaterialModel::kSnow;
    } else if (material_model == "liquid") {
        model = nclr::MaterialModel::kLiquid;
    }


    // Ew
    if (dim.value_or(2) == 2) {
        const auto particles = generate_cubes<2>(cubes, cube_res, args);
        auto sim = std::make_unique<nclr::MPMSimulation<2>>(particles, model, kGridResolution, kDt, E.value_or(1000.0),
                                                            nu.value_or(0.3), gravity.value_or(-100.0));
        std::vector<std::vector<nclr::Particle<2>>> states;
        std::vector<std::vector<nclr::Cell<2>>> cells;
        solve_mpm<2>(sim, steps.value_or(1000), dump, states, cells);
#ifdef NCLR_SOLVER_VIZ
        taichi::GUI gui("Results", kWindowSize, kWindowSize);
        auto &canvas = gui.get_canvas();
        for (const auto &state : states) {
            // Clear background
            canvas.clear(0x112F41);

            // Boundary Condition Box
            canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
            for (const auto &particle : state) {
                // Load the particle
                canvas.circle(taichi::Vector2(particle.x)).radius(2).color(particle.c);
            }

            gui.update();
        }
#endif

        if (dump) {
            unload_particles<2>(material_model.value_or("jelly"), sim, states);
            unload_cells<2>(sim, cells);
        }
    } else {
        auto particles = std::vector<nclr::Particle<3>>{};
        auto sim = std::make_unique<nclr::MPMSimulation<3>>(particles, model, kGridResolution, kDt, E.value_or(1000.0),
                                                            nu.value_or(0.3), gravity.value_or(-100.0));
        /* const auto states = solve_mpm<3>(sim, steps.value_or(1000), dump); */
    }
}
