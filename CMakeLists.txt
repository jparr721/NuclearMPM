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

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/NuclearMPMWarnings.cmake)
include(cmake/NuclearMPMSanitizers.cmake)

set(BUILD_TESTING OFF)
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
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(nuclear_mpm INTERFACE Eigen3::Eigen)
set_target_properties(nuclear_mpm PROPERTIES EXPORT_NAME NuclearMPM)

set(NUCLEAR_MPM_PACKAGE_NEEDS_OPENMP OFF)
if(NUCLEAR_MPM_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nuclear_mpm INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(nuclear_mpm INTERFACE NUCLEAR_MPM_HAS_OPENMP=1)
    set(NUCLEAR_MPM_PACKAGE_NEEDS_OPENMP ON)
  else()
    message(WARNING "NUCLEAR_MPM_ENABLE_OPENMP is ON, but OpenMP was not found; using serial execution")
  endif()
endif()

add_subdirectory(flags)

add_library(nuclear_mpm_cli src/solver_cli.cpp src/snapshot_io.cpp)
target_include_directories(nuclear_mpm_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_cli PUBLIC NuclearMPM::NuclearMPM PRIVATE flags::flags)
nuclear_mpm_enable_warnings(nuclear_mpm_cli)
nuclear_mpm_enable_sanitizers(nuclear_mpm_cli)

add_executable(nuclear_mpm_solver src/solver.cpp)
target_link_libraries(nuclear_mpm_solver PRIVATE nuclear_mpm_cli)
nuclear_mpm_enable_warnings(nuclear_mpm_solver)
nuclear_mpm_enable_sanitizers(nuclear_mpm_solver)

add_library(nuclear_mpm_camera src/camera.cpp)
target_include_directories(nuclear_mpm_camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_camera PUBLIC NuclearMPM::NuclearMPM)
nuclear_mpm_enable_warnings(nuclear_mpm_camera)
nuclear_mpm_enable_sanitizers(nuclear_mpm_camera)

add_library(nuclear_mpm_overlay src/overlay_geometry.cpp)
target_include_directories(nuclear_mpm_overlay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_overlay PUBLIC nuclear_mpm_camera)
nuclear_mpm_enable_warnings(nuclear_mpm_overlay)
nuclear_mpm_enable_sanitizers(nuclear_mpm_overlay)

add_library(nuclear_mpm_session src/simulation_session.cpp)
target_include_directories(nuclear_mpm_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nuclear_mpm_session PUBLIC NuclearMPM::NuclearMPM nuclear_mpm_camera)
nuclear_mpm_enable_warnings(nuclear_mpm_session)
nuclear_mpm_enable_sanitizers(nuclear_mpm_session)

if(NUCLEAR_MPM_BUILD_WORKBENCH)
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
  target_link_libraries(nuclear_mpm_visualizer PRIVATE nuclear_mpm_camera nuclear_mpm_overlay glfw glad_gl_core_41 OpenGL::GL)
  nuclear_mpm_enable_warnings(nuclear_mpm_visualizer)
  nuclear_mpm_enable_sanitizers(nuclear_mpm_visualizer)

  FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.8
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(imgui)

  add_library(
    nuclear_mpm_imgui
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(nuclear_mpm_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(nuclear_mpm_imgui PUBLIC glfw OpenGL::GL)

  add_library(nuclear_mpm_workbench src/simulation_workbench.cpp)
  target_include_directories(nuclear_mpm_workbench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(
    nuclear_mpm_workbench
    PUBLIC nuclear_mpm_session nuclear_mpm_visualizer
    PRIVATE nuclear_mpm_imgui
  )
  nuclear_mpm_enable_warnings(nuclear_mpm_workbench)
  nuclear_mpm_enable_sanitizers(nuclear_mpm_workbench)

  add_executable(nuclear_mpm_example src/example.cpp)
  target_link_libraries(nuclear_mpm_example PRIVATE nuclear_mpm_workbench)
  nuclear_mpm_enable_warnings(nuclear_mpm_example)
  nuclear_mpm_enable_sanitizers(nuclear_mpm_example)
endif()

install(
  TARGETS nuclear_mpm
  EXPORT NuclearMPMTargets
  COMPONENT NuclearMPM
)
install(
  DIRECTORY include/nuclear_mpm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT NuclearMPM
)
configure_package_config_file(
  cmake/NuclearMPMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NuclearMPMConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NuclearMPM
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NuclearMPMConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(
  EXPORT NuclearMPMTargets
  FILE NuclearMPMTargets.cmake
  NAMESPACE NuclearMPM::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NuclearMPM
  COMPONENT NuclearMPM
)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearMPMConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NuclearMPMConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NuclearMPM
  COMPONENT NuclearMPM
)

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

  add_test(
    NAME package_consumer
    COMMAND ${CMAKE_COMMAND}
      -DNUCLEAR_MPM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DNUCLEAR_MPM_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunPackageConsumerTest.cmake
  )
  add_test(
    NAME legacy_headers
    COMMAND ${CMAKE_COMMAND}
      -DNUCLEAR_MPM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckLegacyHeaders.cmake
  )
endif()
