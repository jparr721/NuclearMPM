add_executable(
  nuclear_mpm_tests
  smoke_test.cpp
  math_test.cpp
  grid_test.cpp
  stencil_test.cpp
  particle_sampling_test.cpp
  material_test.cpp
  boundary_test.cpp
  transfer_test.cpp
  simulation_test.cpp
  parallel_test.cpp
  solver_cli_test.cpp
  snapshot_io_test.cpp
  camera_test.cpp
  simulation_session_test.cpp
)
target_link_libraries(
  nuclear_mpm_tests
  PRIVATE NuclearMPM::NuclearMPM nuclear_mpm_cli nuclear_mpm_camera nuclear_mpm_session
          Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(nuclear_mpm_tests TEST_PREFIX "unit:")
