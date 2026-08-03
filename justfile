set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build:
    git submodule update --init --recursive
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

build-workbench:
    git submodule update --init --recursive
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DNUCLEAR_MPM_BUILD_WORKBENCH=ON
    cmake --build build --target nuclear_mpm_example

test:
    git submodule update --init --recursive
    cmake -S . -B build-dev -GNinja -DCMAKE_BUILD_TYPE=Debug -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
    cmake --build build-dev --target nuclear_mpm_tests nuclear_mpm_solver
    ctest --test-dir build-dev --output-on-failure

run: build-workbench
    ./build/nuclear_mpm_example

clean:
    rm -rf build build-dev build-sanitize
