#include "simulation_workbench.h"

#include <exception>
#include <iostream>

int main() {
    try {
        nuclear_mpm::viz::SimulationWorkbench workbench;
        workbench.run();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "NuclearMPM workbench error: " << error.what() << '\n';
        return 1;
    }
}
