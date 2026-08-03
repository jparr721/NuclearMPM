# Camera Drag Direction Fix Design

## Goal

Make workbench camera dragging feel like grabbing the visible scene: dragging right or down moves the scene right or down. Preserve existing scroll zoom/dolly behavior and avoid launching a GUI during verification.

## Root Cause

`Visualizer::Impl::begin_frame()` converts GLFW cursor motion directly into `CameraInput`. It currently negates both orbit deltas. `OrbitCamera` already interprets positive yaw and pitch changes so that the visible scene follows positive screen-space motion, making the extra negation reverse horizontal and vertical orbit controls.

Pan requires different signs because translating the camera and target moves the visible scene in the opposite screen direction. Its current `(-delta_x, +delta_y)` mapping already implements drag-follow behavior.

## Design

Extract cursor conversion into a pure, renderer-independent helper in the camera module. The visualizer will pass cursor displacement, active mouse buttons, and accumulated scroll to this helper.

The mapping is:

| Input | Camera input |
|---|---|
| Rightward left drag (`delta_x > 0`) | Positive horizontal orbit |
| Downward left drag (`delta_y > 0`) | Positive vertical orbit |
| Rightward middle drag | Negative horizontal pan |
| Downward middle drag | Positive vertical pan |
| Upward scroll | Positive dolly/zoom |

The existing sensitivity constants remain unchanged. `OrbitCamera`, `OrthographicCamera`, session persistence, and UI capture filtering remain unchanged.

## Testing

Add headless camera tests for the pure mapping helper:

- right/down left-drag produces positive horizontal/vertical orbit input;
- right/down middle-drag preserves negative-horizontal/positive-vertical pan input;
- scroll scaling remains unchanged;
- inactive buttons produce no drag input;
- mapped orbit and pan deltas move representative projected scene points in the same screen direction as the cursor.

The projection-level assertions compose the mapping helper with the real camera view/projection matrices, so the suite verifies user-visible motion rather than only checking intermediate signs.

Follow red-green TDD: add the regression test first and confirm it fails before adding the helper and delegating visualizer input conversion to it. Then run the bounded camera tests and complete test suite. No interactive workbench or long simulation is required.

## Scope

This fix changes only mouse-drag direction mapping. It does not add inversion preferences, alter camera mathematics, change mouse button assignments, or modify simulation behavior.
