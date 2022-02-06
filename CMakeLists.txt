cmake_minimum_required(VERSION 3.4...3.18)
project(nuclear_mpm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion -Ofast")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-enum-enum-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion")

find_package(Eigen3 CONFIG REQUIRED)
find_package(X11 REQUIRED)

add_executable(${PROJECT_NAME} nclr.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen ${X11_LIBRARIES})
