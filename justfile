set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build:
    git submodule update --init --recursive
    mkdir -p build
    cd build && cmake -GNinja ..
    cd build && ninja

run: build
    ./build/nuclear_mpm_example

clean:
    rm -rf build
