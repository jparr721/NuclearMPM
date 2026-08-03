cmake_minimum_required(VERSION 3.16)
project(nuclear_mpm_package_consumer LANGUAGES CXX)

find_package(NuclearMPM CONFIG REQUIRED)

add_executable(nuclear_mpm_package_consumer main.cpp)
target_link_libraries(nuclear_mpm_package_consumer PRIVATE NuclearMPM::NuclearMPM)
