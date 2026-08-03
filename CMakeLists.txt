cmake_minimum_required(VERSION 3.16...3.31)
project(nuclear_mpm)

set(PROJECT_NAME_EXAMPLE ${PROJECT_NAME}_example)
set(PROJECT_NAME_SOLVER ${PROJECT_NAME}_solver)
set(PROJECT_NAME_VISUALIZER ${PROJECT_NAME}_visualizer)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast -pthread")

include(FetchContent)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(eigen glfw)

add_library(glad_gl_core_41 STATIC vendor/glad/src/gl.c)
target_include_directories(glad_gl_core_41 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad/include)
find_package(OpenGL REQUIRED)

if (WITH_NCLR_DEBUG)
  add_compile_definitions(NCLR_DEBUG)
endif()

add_subdirectory(flags)

add_library(${PROJECT_NAME_VISUALIZER} src/visualizer.cpp)
target_include_directories(${PROJECT_NAME_VISUALIZER} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME_VISUALIZER} PRIVATE glfw glad_gl_core_41 OpenGL::GL)

add_executable(${PROJECT_NAME_EXAMPLE} src/example.cpp)
target_link_libraries(${PROJECT_NAME_EXAMPLE} PRIVATE Eigen3::Eigen ${PROJECT_NAME_VISUALIZER})

add_executable(${PROJECT_NAME_SOLVER} src/solver.cpp)
target_link_libraries(${PROJECT_NAME_SOLVER} PRIVATE Eigen3::Eigen flags)

if (WITH_NCLR_SOLVER_VIZ)
  target_compile_definitions(${PROJECT_NAME_SOLVER} PRIVATE NCLR_SOLVER_VIZ)
  target_link_libraries(${PROJECT_NAME_SOLVER} PRIVATE ${PROJECT_NAME_VISUALIZER})
endif()

option(NCLR_BUILD_TESTS "Build NuclearMPM smoke tests" ON)
if (NCLR_BUILD_TESTS)
  enable_testing()

  set(PROJECT_NAME_EXAMPLE_3D_COMPILE ${PROJECT_NAME_EXAMPLE}_3d_compile)
  add_library(${PROJECT_NAME_EXAMPLE_3D_COMPILE} OBJECT EXCLUDE_FROM_ALL src/example.cpp)
  target_compile_definitions(${PROJECT_NAME_EXAMPLE_3D_COMPILE} PRIVATE NCLR_EXAMPLE_DIMENSION=3)
  target_link_libraries(${PROJECT_NAME_EXAMPLE_3D_COMPILE} PRIVATE Eigen3::Eigen ${PROJECT_NAME_VISUALIZER})

  add_test(
    NAME example_3d_compile
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${PROJECT_NAME_EXAMPLE_3D_COMPILE}
  )
  add_test(NAME solver_help COMMAND $<TARGET_FILE:${PROJECT_NAME_SOLVER}> --help)
  set_tests_properties(solver_help PROPERTIES PASS_REGULAR_EXPRESSION "NuclearMPM headless solver")
  add_test(
    NAME solver_one_step
    COMMAND $<TARGET_FILE:${PROJECT_NAME_SOLVER}> --steps 1 --cube0-x 0.4 --cube0-y 0.6
  )
  set_tests_properties(solver_one_step PROPERTIES PASS_REGULAR_EXPRESSION "Simulation done")
endif()
