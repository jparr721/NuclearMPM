cmake_minimum_required(VERSION 3.4...3.18)
project(nuclear_mpm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 CONFIG REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} nclr.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${PROJECT_NAME}
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
