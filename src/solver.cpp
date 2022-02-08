#include "nclr.h"
#include <Eigen/Dense>
#include <cstdint>
#include <flags.h>
#include <memory>

// How many dimensions to run the sim in (2 or 3)
constexpr int kDimension = 2;

// The resolution of the cube meshes.
constexpr int kResolution = 50;

// Window
constexpr int kWindowSize = 800;

// Timestep size
constexpr nclr::real dt = 1e-4f;

// Frame draw interval
constexpr nclr::real frame_dt = 1e-3f;

// The color to paint the points
constexpr int color = 0xED553B;

int main() {}
