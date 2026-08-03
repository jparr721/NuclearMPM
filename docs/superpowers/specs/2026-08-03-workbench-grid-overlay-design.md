# Workbench Grid and Boundary Overlay Design

## Goal

Restore the workbench's expected free-fall behavior and make the MPM grid-to-boundary relationship visible.

## Behavior

- Keep both 2D and 3D sessions paused at startup.
- Change the default workbench grid from 8 to 64 cells per axis.
- Add `Show grid` and `Show boundary guides` controls, enabled by default.
- Visualization settings affect rendering only and do not rebuild or mutate a simulation.

## Rendering

The active render scene carries the running simulation's cell count and per-face boundary configuration. Staged settings do not affect the overlay until the user resets the simulation.

The visualizer uses wire geometry:

- In 2D, draw all grid lines across the unit domain.
- In 3D, draw grid lines on the lower Y, lower X, and lower Z domain faces rather than filling the volume.
- Draw per-face boundary influence guides with colors that distinguish slip, sticky, and outflow behavior.
- Retain the existing unit-domain outline and particle rendering.

For slip and sticky faces, the guide marks where a quadratic particle stencil first reaches a constrained boundary node. With `N` cells and `L` boundary layers, the lower guide is `(L + 0.5) / N` and the upper guide is `1 - (L + 0.5) / N`. For outflow faces, the guide uses the configured deletion plane at `outflow_deletion_layers / N` from the domain edge.

## Testing

- Verify default sessions remain paused and use 64 cells.
- Verify unpausing the default scene produces sustained gravitational free fall before boundary contact.
- Verify render-scene overlay metadata comes from the active simulation and changes only after reset.
- Verify generated 2D and 3D wire geometry has the expected line counts, coordinates, and boundary-mode colors without requiring an OpenGL context.

## Non-goals

- Rendering every internal 3D cell.
- Movable grid slices.
- Changing the core boundary algorithm or normalized unit domain.
- World-scale sparse/chunked MPM support.
