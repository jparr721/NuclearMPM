cmake_minimum_required(VERSION 3.16...3.31)
project(nuclear_mpm VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NUCLEAR_MPM_BUILD_TESTS "Build NuclearMPM tests" ON)
option(NUCLEAR_MPM_BUILD_WORKBENCH "Build the OpenGL workbench" ON)
option(NUCLEAR_MPM_ENABLE_OPENMP "Enable the bounded OpenMP backend" OFF)
option(NUCLEAR_MPM_ENABLE_SANITIZERS "Enable supported sanitizers" OFF)
option(WITH_NCLR_SOLVER_VIZ "Build legacy solver playback support" OFF)

include(FetchContent)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen)

add_library(nuclear_mpm INTERFACE)
add_library(NuclearMPM::NuclearMPM ALIAS nuclear_mpm)
target_compile_features(nuclear_mpm INTERFACE cxx_std_17)
target_include_directories(
  nuclear_mpm
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nuclear_mpm INTERFACE Eigen3::Eigen)

if(NUCLEAR_MPM_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nuclear_mpm INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(nuclear_mpm INTERFACE NUCLEAR_MPM_HAS_OPENMP=1)
  else()
    message(WARNING "NUCLEAR_MPM_ENABLE_OPENMP is ON, but OpenMP was not found; using serial execution")
  endif()
endif()

add_subdirectory(flags)

add_library(nuclear_mpm_cli src/solver_cli.cpp src/snapshot_io.cpp)
target_include_directories(nuclear_mpm_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_cli PUBLIC NuclearMPM::NuclearMPM PRIVATE flags::flags)

add_executable(nuclear_mpm_solver src/solver.cpp)
target_link_libraries(nuclear_mpm_solver PRIVATE nuclear_mpm_cli)

add_library(nuclear_mpm_camera src/camera.cpp)
target_include_directories(nuclear_mpm_camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_camera PUBLIC NuclearMPM::NuclearMPM)

if(NUCLEAR_MPM_BUILD_WORKBENCH OR WITH_NCLR_SOLVER_VIZ)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(glfw)

  add_library(glad_gl_core_41 STATIC vendor/glad/src/gl.c)
  target_include_directories(glad_gl_core_41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad/include)
  find_package(OpenGL REQUIRED)

  add_library(nuclear_mpm_visualizer src/visualizer.cpp)
  target_include_directories(nuclear_mpm_visualizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(nuclear_mpm_visualizer PRIVATE nuclear_mpm_camera glfw glad_gl_core_41 OpenGL::GL)

  if(NUCLEAR_MPM_BUILD_WORKBENCH)
    add_executable(nuclear_mpm_example src/example.cpp)
    target_link_libraries(nuclear_mpm_example PRIVATE NuclearMPM::NuclearMPM nuclear_mpm_visualizer)
  endif()

  if(WITH_NCLR_SOLVER_VIZ)
    target_compile_definitions(nuclear_mpm_solver PRIVATE NCLR_SOLVER_VIZ)
    target_link_libraries(nuclear_mpm_solver PRIVATE nuclear_mpm_visualizer)
  endif()
endif()

if(NUCLEAR_MPM_BUILD_TESTS)
  enable_testing()

  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.8.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Catch2)
  add_subdirectory(tests)

  add_test(NAME solver_help COMMAND $<TARGET_FILE:nuclear_mpm_solver> --help)
  set_tests_properties(solver_help PROPERTIES PASS_REGULAR_EXPRESSION "NuclearMPM solver")
  add_test(
    NAME solver_2d_one_step
    COMMAND $<TARGET_FILE:nuclear_mpm_solver>
      --dimension 2 --material solid --box-min 0.4,0.4 --box-max 0.6,0.6
      --points-per-axis 2,2 --cells 8 --duration 0.0001 --max-dt 0.0001
  )
  add_test(
    NAME solver_3d_one_step
    COMMAND $<TARGET_FILE:nuclear_mpm_solver>
      --dimension 3 --material solid --box-min 0.4,0.4,0.4 --box-max 0.6,0.6,0.6
      --points-per-axis 2,2,2 --cells 8 --duration 0.0001 --max-dt 0.0001
  )
  set_tests_properties(solver_2d_one_step solver_3d_one_step
    PROPERTIES PASS_REGULAR_EXPRESSION "Simulation complete")
endif()
