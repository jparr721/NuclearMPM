# Justfile Build and Run Design

## Goal

Add a root-level `justfile` that provides simple commands for configuring, building, running, and cleaning NuclearMPM while retaining the project's documented Ninja workflow. Make Eigen self-contained so developers do not need a system Eigen installation.

## Recipes

- `build`: initialize repository submodules, create `build/` if needed, configure from inside it with `cmake -GNinja ..`, and compile with `ninja`.
- `run`: invoke `build` first, then launch `./build/nuclear_mpm_example`.
- `clean`: remove the generated `build/` directory.

## Dependencies

CMake uses `FetchContent` to download pinned Eigen 3.4.0 and GLFW 3.4 sources into the build tree. Eigen is not required from a system package manager. The renderer uses system OpenGL and the generated GLAD loader under `vendor/glad`; it has no direct X11 linkage. The existing `flags` dependency remains a Git submodule.

## Behavior

Recipes use Bash and stop immediately when a command fails. Repeated builds are supported by using `mkdir -p build`; CMake reconfiguration and Ninja incremental compilation remain responsible for keeping build output current. The build recipe makes fresh clones usable by initializing submodules before configuration. The run recipe intentionally selects the GUI example rather than the headless solver.

## Verification

Confirm that Just recognizes all recipes, CMake downloads Eigen without a system Eigen package, `just build` produces `build/nuclear_mpm_example`, and repeated builds succeed. Confirm that `just run` resolves to the example executable. If the host cannot display an OpenGL window, command expansion and executable linkage are sufficient to validate the recipe while documenting the environment limitation.
