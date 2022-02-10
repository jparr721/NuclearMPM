cmake_minimum_required(VERSION 3.12...3.18)
project(nuclear_mpm)

set(PROJECT_NAME_EXAMPLE ${PROJECT_NAME}_example)
set(PROJECT_NAME_SOLVER ${PROJECT_NAME}_solver)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast -pthread")

find_package(Eigen3 CONFIG REQUIRED)
find_package(X11 REQUIRED)

if (WITH_NCLR_SOLVER_VIZ)
  add_definitions(-DNCLR_SOLVER_VIZ)
endif()

if (WITH_NCLR_DEBUG)
  add_defitions(-DNCLR_DEBUG)
endif()

add_subdirectory(flags)

add_executable(${PROJECT_NAME_EXAMPLE} src/example.cpp)
target_link_libraries(${PROJECT_NAME_EXAMPLE} PRIVATE Eigen3::Eigen ${X11_LIBRARIES})

add_executable(${PROJECT_NAME_SOLVER} src/solver.cpp)
target_link_libraries(${PROJECT_NAME_SOLVER} PRIVATE Eigen3::Eigen ${X11_LIBRARIES} flags)
