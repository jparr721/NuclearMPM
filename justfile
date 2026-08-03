set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build:
    git submodule update --init --recursive
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DNUCLEAR_MPM_BUILD_TESTS=OFF
    cmake --build build

build-workbench:
    git submodule update --init --recursive
    cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DNUCLEAR_MPM_BUILD_TESTS=OFF -DNUCLEAR_MPM_BUILD_WORKBENCH=ON
    cmake --build build --target nuclear_mpm_example

test:
    git submodule update --init --recursive
    cmake -S . -B build-dev -GNinja -DCMAKE_BUILD_TYPE=Debug -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF
    cmake --build build-dev --target nuclear_mpm_tests nuclear_mpm_solver
    ctest --test-dir build-dev --output-on-failure

test-sanitize:
    git submodule update --init --recursive
    cmake -S . -B build-sanitize -GNinja -DCMAKE_BUILD_TYPE=Debug -DNUCLEAR_MPM_BUILD_TESTS=ON -DNUCLEAR_MPM_BUILD_WORKBENCH=OFF -DNUCLEAR_MPM_ENABLE_SANITIZERS=ON
    cmake --build build-sanitize --target nuclear_mpm_tests nuclear_mpm_solver
    ctest --test-dir build-sanitize --output-on-failure --timeout 60 -R '^(solver_|unit_all_sanitized|package_consumer|legacy_headers)'

run: build-workbench
    ./build/nuclear_mpm_example

clean:
    rm -rf build build-dev build-sanitize
