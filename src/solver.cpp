#include "solver_cli.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            std::cout << nuclear_mpm::cli::help_message();
            return EXIT_SUCCESS;
        }
        return nuclear_mpm::cli::run_solver(nuclear_mpm::cli::parse_solver_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "NuclearMPM solver error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
