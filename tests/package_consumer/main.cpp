#include <nuclear_mpm/nuclear_mpm.hpp>

int main() {
    nuclear_mpm::Grid<2> grid(4);
    nuclear_mpm::Particle<3> particle;
    particle.mass = 1.0F;
    particle.reference_volume = 1.0F;
    nuclear_mpm::validate_particle(particle, 0);
    return grid.node_count() == 25 ? 0 : 1;
}
