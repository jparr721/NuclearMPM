# Game Physics CPU Backend Design

## Purpose

Turn NuclearMPM into a plug-in-style real-time deformable physics engine, beginning with a custom CPU parallel backend. The public API should resemble a conventional game physics SDK: callers create an engine and scene, add deformable bodies, submit a timestep, and fetch published results. Internal CFL substeps, task scheduling, tile exchange, and synchronization remain hidden.

The first performance workload is one million fixed-corotated solid particles on a dense `128³` grid on Apple Silicon. A complete `simulate(1/30)` call has a stretch budget of 16 ms. This is a measured optimization target, not an up-front guarantee.

## Scope

This phase includes:

- an engine-owned persistent CPU task runtime;
- an asynchronous scene simulation/fetch API;
- a deterministic tiled CPU MPM backend;
- parallel particle binning, P2G, grid update, G2P, and diagnostics;
- bounded, reusable scratch memory;
- per-phase timing and budget-overrun telemetry;
- an opt-in million-particle benchmark.

This phase does not implement Metal compute, networking, rigid-body dynamics, chunk streaming, surface meshing, or a fracture/damage constitutive model. It preserves extension points for those systems. Fixed-corotated cubes can deform and separate where particle support disconnects, but convincing breakage requires a later damage/fracture phase.

## Public Engine API

The game-facing ownership model is:

```cpp
nuclear_mpm::PhysicsEngine engine(engine_config);
auto scene = engine.create_scene(scene_config);

auto body = scene->create_deformable_cube(cube_description);
scene->simulate(1.0F / 30.0F);
scene->fetch_results();
```

`PhysicsEngine` owns the task runtime and may own multiple scenes. `PhysicsScene::simulate(dt)` submits a complete update and returns after scheduling it. `fetch_results()` waits for completion, propagates contextual failures, and publishes a stable result snapshot. A scene cannot accept a second simulation while one is in flight.

Initial public types:

- `EngineConfig`: worker count (`0` selects available hardware concurrency, including the calling thread) and a 64 MiB default scratch-memory budget.
- `SceneConfig`: grid resolution, material table, boundaries, timestep/CFL controls, and frame budget.
- `DeformableCubeDescription`: bounds, particle counts, density, material identifier, initial velocity, and user tag.
- `BodyHandle`: generation-checked opaque identifier.
- `SceneStats`: total wall time, phase timings, internal substeps, active particles, active tiles, scratch high-water mark, and budget-overrun count.

The initial game-engine scene API is three-dimensional. The existing `Simulation<Dim>` continues to support both 2D and 3D and remains the deterministic serial numerical reference while the engine API is introduced. Bounded equivalence tests compare the CPU backend against it.

## Backend Boundary

The engine invokes a backend through a coarse scene-step boundary rather than virtual calls inside hot loops. The initial `CpuTiledBackend` owns CPU-specific storage and execution. A future `MetalMpmBackend` can implement the same scene lifecycle and result publication contract without changing game code.

```text
PhysicsEngine
├── TaskRuntime
└── PhysicsScene
    ├── Scene commands and body handles
    ├── Published result buffers
    └── SimulationBackend
        ├── CpuTiledBackend
        └── MetalMpmBackend (future)
```

## Persistent Task Runtime

`TaskRuntime` uses C++17 threads owned for the lifetime of `PhysicsEngine`. It does not spawn threads per simulation phase. The calling thread participates in work while waiting.

The first runtime primitive is a bounded `parallel_for` over stable task identifiers. Workers dynamically claim task identifiers so Apple performance and efficiency cores remain utilized without manual affinity. Each task writes only to its assigned output or reduction slot; scheduling order therefore does not determine numerical reduction order.

Runtime requirements:

- no heap allocation in steady-state dispatch;
- reusable job descriptors and worker-local scratch;
- condition-variable sleep while idle rather than busy spinning between frames;
- cooperative shutdown in the engine destructor;
- first-failure capture with phase and task context;
- no partial result publication after failure;
- nested parallel dispatch is rejected initially to avoid worker starvation and ambiguous ownership.

Apple-specific QoS tuning may be added behind the runtime implementation, but the task API remains portable C++.

## Particle and Grid Storage

The CPU backend uses structure-of-arrays particle blocks rather than one large array of Eigen-heavy particle structs. Positions, velocities, affine matrices, deformation matrices, mass, volume, material identifiers, and user tags occupy separate aligned arrays. Stable particle identifiers are distinct from storage indices so compaction does not invalidate body handles.

The grid is split into `16³` cell tiles. Only active tiles allocate or reset simulation storage. Tiles are stored in deterministic coordinate order and maintain a lookup from tile coordinate to dense active index.

Particles are binned by the tile containing their quadratic-stencil base coordinate. Each P2G task owns one particle bin and a tile-local mass/momentum accumulator with the halo required by the three-node quadratic stencil. Scratch is allocated once and reused. It is limited by the configured budget rather than `worker_count × global_grid_size`.

If all active tile accumulators do not fit the scratch budget, P2G runs deterministic tile batches. After each batch, one merge task per affected canonical tile reads contributing accumulators in lexicographic tile order before that scratch is reused. Batches execute in stable tile order. A canonical tile has one writer within each merge barrier, producing deterministic reductions within floating-point tolerance without global floating-point atomics.

## Simulation Phase Graph

Each internal CFL substep runs these barriers:

1. **Classify and bin particles** — validate support, count particles per tile, prefix-sum counts, and scatter stable particle indices.
2. **Prepare active tiles** — reset canonical grid data and reusable tile-local accumulators.
3. **Particle-to-grid** — process particle bins into private tile accumulators.
4. **Deterministic halo merge** — one task per canonical tile combines contributors in fixed order.
5. **Grid update** — normalize momentum, apply gravity and boundary policies per tile.
6. **Grid-to-particle** — update independent particle ranges from the read-only canonical grid.
7. **Diagnostics and outflow** — write per-task reduction slots, combine slots by stable task identifier, and compact removals deterministically.

`simulate(dt)` selects internal substeps using the existing CFL policy and schedules all required phases. The engine never drops required substeps merely to satisfy the frame budget. `SceneStats` records overruns so later work can introduce sleeping regions, level of detail, softer gameplay materials, or a Metal backend deliberately.

## Game-System Interaction

Game code submits scene changes at simulation boundaries through commands rather than mutating live particle buffers. Initial commands create or remove bodies, change gravity, apply forces/impulses, and update material or boundary parameters.

`fetch_results()` publishes body summaries, particle positions/tags needed by rendering, diagnostics, and generated events. The CPU backend may double-buffer lightweight published position/event data; it does not duplicate the complete million-particle simulation state solely for presentation.

Future rigid bodies, damage, and chunk systems integrate as additional phase inputs/outputs. A future Metal backend will keep particle/render buffers GPU-resident and return only reduced gameplay events to the CPU.

## Determinism and Failure Behavior

Parallel results must agree with the serial reference within documented floating-point tolerances. Stable bin ordering, one-writer output ownership, fixed halo merge order, and task-index reduction slots make results independent of worker scheduling.

Worker exceptions are captured with phase and task identifiers. `fetch_results()` rethrows a contextual `SimulationError`, leaves the prior published snapshot intact, and marks the failed scene as requiring reset or explicit recovery.

## Performance Measurement

Add an opt-in benchmark executable that reports:

- particle classification/binning time;
- P2G time;
- merge time;
- grid update time;
- G2P time;
- diagnostics/compaction time;
- total time per internal substep and per `simulate(dt)` call;
- effective particle updates per second;
- active tile count and scratch-memory high-water mark.

The benchmark scales through small presets before the explicit target preset:

- 10,000 particles for smoke measurement;
- 100,000 particles for iteration;
- 1,000,000 fixed-corotated particles on `128³` cells for the target measurement.

The million-particle preset is never registered with CTest and is never run as unattended routine verification. Performance claims must name the Apple hardware, build type, particle/grid/material configuration, number of internal substeps, and measured phase timings.

## Migration Strategy

1. Add the runtime and test its lifecycle, exact-once execution, failure propagation, and stable reductions.
2. Introduce backend-neutral engine/scene ownership while delegating to the current serial simulation.
3. Add particle SoA adapters and verify round trips against existing particle state.
4. Add active tile classification and deterministic particle binning.
5. Implement tiled P2G and fixed-order halo merging against serial transfer invariants.
6. Parallelize grid update, G2P, diagnostics, and outflow compaction.
7. Remove the OpenMP implementation and package dependency after bounded parity tests pass.
8. Add the opt-in benchmark and optimize measured bottlenecks without weakening numerical tests.

At every migration step, the serial implementation remains executable and bounded tests remain the correctness gate.
