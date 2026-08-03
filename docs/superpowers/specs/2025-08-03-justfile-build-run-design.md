# Justfile Build and Run Design

## Goal

Add a root-level `justfile` that provides simple commands for configuring, building, running, and cleaning NuclearMPM while retaining the project's documented Ninja workflow.

## Recipes

- `build`: create `build/` if needed, configure from inside it with `cmake -GNinja ..`, and compile with `ninja`.
- `run`: invoke `build` first, then launch `./build/nuclear_mpm_example`.
- `clean`: remove the generated `build/` directory.

## Behavior

Recipes use Bash and stop immediately when a command fails. Repeated builds are supported by using `mkdir -p build`; CMake reconfiguration and Ninja incremental compilation remain responsible for keeping build output current. The run recipe intentionally selects the GUI example rather than the headless solver.

## Verification

Confirm that Just recognizes all recipes, that `just build` produces `build/nuclear_mpm_example`, and that `just run` reaches the example executable. If the host cannot display the X11 GUI, executable startup or linkage is sufficient to validate the recipe while documenting the environment limitation.
