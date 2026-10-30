# 3GPP TR 37.885 Urban V2X Channel Implementation Plan

## 1. Purpose

This plan adds a reproducible, standards-based urban-grid V2X scene and channel
simulator to Channel Dataset. The target is the useful middle ground between a
single abstract V2V link and a map-based or ray-traced city:

- the road grid, lanes, intersections, sidewalks, vehicles, mobility, and link
  states are explicit;
- the channel remains the statistical channel defined by 3GPP TR 37.885 rather
  than a deterministic building-reflection model;
- the exact reference scene and controlled randomized variations are both
  supported, with provenance that makes the distinction unambiguous;
- the output remains a reusable `CIRBatch` and works with the existing time and
  OFDM transforms, signal operators, metrics, datasets, and backends.

The normative baseline is [3GPP TR 37.885 V15.3.0, Release 15](https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3209).
The complete V15.3.0 text is available from the [3GPP partner publication](https://atisorg.s3.amazonaws.com/archive/3gpp-documents/Rel15/ATIS.3GPP.37.885.V1530.pdf).
V15.3.0 is used because it is the final listed Release-15 version and includes
the approved V2X antenna/path-loss correction in CR 0006.

## 2. Acceptance boundary

### 2.1 Included

The first release includes:

1. The Annex A urban reference grid:
   - 433 m by 250 m intersection spacing;
   - two 3.5 m lanes in each direction on every street;
   - 3 m sidewalk on each side;
   - 20 m total street width;
   - a minimum 1299 m by 750 m simulation area.
2. Urban mobility Options A and B from clause 6.1.2.
3. Vehicle types 1, 2, and 3, including dimensions and antenna heights.
4. Standard vehicle dropping, gaps, lane speeds, and intersection turns.
5. A 100 ms mobility and link-condition update interval.
6. Urban V2V sidelink states LOS, NLOSv, and building-obstructed NLOS.
7. Urban V2V path loss, spatially correlated shadow fading, and NLOSv vehicle
   blockage loss.
8. The urban LOS, NLOSv, and NLOS fast-fading parameter sets.
9. Dual-mobility Doppler and the delayed-path moving-scatterer term.
10. The fixed urban LOS, NLOSv, and NLOS CDL profiles in clause 6.2.3.1.
11. Below-6-GHz operation with the standard 6 GHz sidelink evaluation point.
12. Standard vehicle antenna Option 1 at 6 GHz, using the repository antenna
    array and field-pattern contracts.
13. Exact replay, NumPy/PyTorch execution, persistence, diagnostics, examples,
    and Sphinx documentation.
14. An optional controlled-variation mode for research datasets.

The initial public model keys are:

- `wireless/tr37885_urban_system`
- `wireless/tr37885_urban_cdl`

### 2.2 Deferred

The following are deliberately outside this release:

- highway scenarios;
- real-world maps, GIS/OSM import, and external traffic simulators;
- three-dimensional building meshes, facade materials, ray tracing,
  deterministic reflection, and diffraction;
- above-6-GHz 30/63 GHz antenna patterns and oxygen absorption;
- macro/micro base-station links and cellular system layout;
- pedestrian and in-vehicle cellular UE population models;
- MAC, scheduling, traffic generation, PRR/PIR, packet throughput, and network
  performance simulation;
- lane changing, collision avoidance, traffic lights, congestion feedback, and
  car-following dynamics;
- arbitrary user-authored road networks.

V2R and R2R may later reuse the V2V path-loss implementation, as allowed by the
TR, but they are not required for the first acceptance gate. Highway and
above-6-GHz work should be separate extensions rather than hidden options in
the urban baseline.

## 3. Standard versus controlled variation

Every scene configuration has an explicit `mode`:

### `standard`

All normative values are locked to the selected TR profile. The seed may vary,
which randomizes:

- initial vehicle gaps and positions;
- vehicle types where the selected mobility option defines a mixture;
- Option-A intersection choices;
- LOS/NLOSv draws for same-street links;
- blockage, shadowing, LSP, ray, phase, and moving-scatterer draws.

The standard mode must reject attempts to change lane width, block spacing,
street width, speeds, fleet mixture, turn probabilities, or update interval.

### `controlled_variation`

This mode may override a documented subset:

- grid extent while preserving the rectangular street construction;
- mean gap or a density multiplier;
- vehicle-type probabilities;
- per-lane speeds or a global speed multiplier;
- turn probabilities;
- mobility update interval;
- carrier frequency within the validated below-6-GHz applicability range;
- same-street state policy and boundary policy.

The realization metadata records every resolved value and includes
`standard_baseline: false`. Variations must never be described as an exact
3GPP evaluation case.

The urban boundary behavior is not silently inferred. `exit` is the reference
finite-area policy; `respawn` and periodic tiling are explicit controlled
variations. The implementation inventory in Phase 0 will record any additional
source guidance before this policy is finalized.

## 4. Reuse and new work

### 4.1 Reused infrastructure

The following existing capabilities are directly reusable:

- `ArrayBackend`, NumPy/PyTorch backends, and `RandomStream`;
- `ExecutionContext`, configuration envelopes, hashes, registry, and resource
  loading;
- `TopologyBatch` as the per-snapshot view passed to common geometry and
  coefficient kernels;
- `LinkGeometry`, antenna arrays, polarized field evaluation, matrix roots,
  and correlation utilities;
- `LargeScaleParameters`, `RayBatch`, `ClusterBatch`, and `CIRBatch`;
- the standard-neutral portions of `ScenarioRayGenerator` and
  `SystemLevelCoefficientGenerator`;
- `cir_to_time`, `cir_to_ofdm`, signal application, normalization, and AWGN;
- spatial fields, metrics, persistence, streaming, replay, and corruption
  detection;
- Sphinx API generation and the existing NumPy/PyTorch CI matrix.

### 4.2 Required extensions

New work is required for:

- a time-indexed vehicle fleet rather than independent rectangular endpoints;
- typed streets, lanes, intersections, sidewalks, and building blocks;
- lane-constrained dropping and motion;
- stable vehicle IDs and route decisions;
- same-street/building-obstruction classification;
- LOS/NLOSv persistence and state transitions at mobility snapshots;
- TR 37.885 vehicle blockage loss;
- three-state large-scale and ray generation;
- spatial state continuity across successive snapshots;
- the additional moving-scatterer Doppler term;
- V2X-specific fixed CDL resources and assembly;
- V2X realization/timeline persistence and diagnostics.

The generic `TopologyBatch` must not be changed to carry roads or trajectories.
A V2X trajectory creates a validated `TopologyBatch` view for a selected
snapshot and endpoint set.

## 5. Target package and resources

```text
src/channel_dataset/channels/wireless/tr37885/
  __init__.py
  config.py                 Strict model and scene configurations
  types.py                  Scene, trajectory, state, and realization contracts
  resources.py              Versioned resource loading and validation
  grid.py                   Streets, lanes, intersections, and building blocks
  traffic.py                Vehicle dropping and mobility Options A/B
  topology.py               Snapshot conversion and link selection
  link_state.py             LOS/NLOSv/NLOS classification and persistence
  blockage.py               Vehicle-blockage attenuation
  large_scale.py            Path loss, shadowing, and composition
  fading.py                 Three-state stochastic fast fading
  doppler.py                Dual mobility and moving scatterers
  cdl.py                    Fixed urban CDL models
  antennas.py               Standard vehicle antenna profiles
  system.py                 End-to-end model assembly
  simulation.py             Multi-snapshot orchestration
  dataset.py                Checksummed persistence and replay
  diagnostics.py            Optional immutable diagnostics

src/channel_dataset/resources/standards/wireless/tr_37_885/v15_3/
  source.yaml
  urban_grid.yaml
  vehicle_types.yaml
  urban_mobility.yaml
  urban_pathloss.yaml
  urban_fast_fading.yaml
  urban_cdl_los.yaml
  urban_cdl_nlosv.yaml
  urban_cdl_nlos.yaml
  vehicle_antenna_6ghz.yaml

src/channel_dataset/resources/schema/
  tr37885-urban-scene-v1.schema.json
  tr37885-urban-system-v1.schema.json
  tr37885-urban-cdl-v1.schema.json

configs/schema/
configs/wireless/tr37885/
  urban_option_a.yaml
  urban_option_b.yaml
  urban_controlled_variation.yaml

tests/unit/wireless/tr37885/
tests/integration/wireless/tr37885/
tests/statistical/wireless/tr37885/
tests/reference/wireless/tr37885/
```

## 6. Public interfaces and array contracts

### 6.1 Scene and trajectory

```python
TR37885UrbanGridConfig(...)
TR37885TrafficConfig(option="A", ...)
TR37885UrbanSystemConfig(...)
TR37885UrbanCDLConfig(...)

TR37885UrbanGrid(config)
TR37885UrbanGridSimulator(grid, traffic)
V2XSceneSamplingRequest(...)
V2XSceneTrajectory(...)
```

`V2XSceneTrajectory` is immutable and uses:

- positions, velocities, and orientations:
  `[batch, vehicle, mobility_time, xyz_or_ypr]`;
- active mask, lane ID, street ID, and route state:
  `[batch, vehicle, mobility_time]`;
- vehicle type, dimensions, and antenna height:
  `[batch, vehicle]` or `[batch, vehicle, dimension]`;
- mobility times: `[mobility_time]`.

Stable integer vehicle IDs are part of the contract. Random stream labels use
those IDs, not array positions, so reordering or selecting vehicles does not
change their routes or channel draws.

### 6.2 Link conditions

```python
V2XLinkConditionBatch(...)
V2XLinkState(...)
classify_urban_links(...)
```

The condition codes are `LOS`, `NLOSV`, `NLOS`, and `INVALID`, on
`[batch, rx, tx, mobility_time]`. Diagnostics include:

- same-street mask;
- building-obstruction mask;
- standardized LOS probability;
- selected blocker type and height;
- optional geometric blocker vehicle ID;
- transition and episode indices.

In strict baseline mode, a same-street LOS/NLOSv draw is held fixed as required
by the baseline. Transitions between same-street states and building NLOS are
evaluated every 100 ms. An optional dynamic LOS/NLOSv policy is available only
as a labeled controlled variation.

### 6.3 Channel sampling and realization

```python
V2XChannelSamplingRequest(
    sampling,
    scene,
    snapshot_index,
    tx_vehicle_indices,
    rx_vehicle_indices,
    state=None,
)

TR37885UrbanSystemChannel(config).realize(request)
TR37885UrbanCDLChannel(config).realize(request)
TR37885UrbanGridSimulator.realize_sequence(...)
```

`V2XChannelRealization` contains:

- `cir: CIRBatch`;
- selected per-snapshot `TopologyBatch`;
- `link_conditions: V2XLinkConditionBatch` view;
- decomposed `V2XLargeScaleComponents`;
- immutable `V2XChannelState` for the next mobility snapshot;
- optional diagnostics and full resource provenance.

One realization covers a waveform window whose propagation state is fixed at
one mobility snapshot. `realize_sequence` returns a sequence of realizations
and carries immutable state between them. This avoids pretending that
time-varying path delays fit the existing static-delay `CIRBatch` contract.

## 7. Randomness and continuity

The root seed derives labeled child streams from stable scene identities:

```text
tr37885.scene.<batch>.lane.<lane_id>.gap.<ordinal>
tr37885.vehicle.<vehicle_id>.type
tr37885.vehicle.<vehicle_id>.intersection.<intersection_id>.turn
tr37885.link.<min_id>.<max_id>.episode.<episode>.condition
tr37885.link.<...>.shadowing
tr37885.link.<...>.lsp
tr37885.link.<...>.cluster.<cluster>.ray.<ray>.phase
tr37885.link.<...>.cluster.<cluster>.ray.<ray>.scatterer
```

Required guarantees:

- adding diagnostics does not alter physical draws;
- vehicle array order does not alter a vehicle's route;
- link subset or evaluation order does not alter an existing link;
- system and CDL models do not share accidental streams;
- replay is exact within a backend;
- NumPy and PyTorch need deterministic mathematical parity but not identical
  stochastic samples.

Shadow fading and LSPs evolve through distance-correlated fields. Cluster and
ray identities persist while a link remains in the same propagation episode.
Entering building NLOS or returning to a same-street state starts a documented
new episode and derives a new stream without depending on draw order.

## 8. Implementation phases and gates

### Phase 0 — Standards and resource baseline

- Acquire and checksum the official `37885-f30` source archive.
- Inventory clauses 6.1.1–6.1.4, 6.2, 6.2.1–6.2.3.1, Annex A, CR 0006,
  and all referenced TR 36.885/TR 38.901 procedures.
- Pin TR 36.885 V14.0.0 and the applicable Release-15 TR 38.901 procedure
  revision; compare reused equations field by field with repository V16.1.0.
- Transcribe the grid, vehicles, mobility, path loss, fast-fading table, CDL
  profiles, and antenna parameters into independent YAML resources.
- Record schema version, source document/version, clause/table, units, source
  URL, checksum, transcription date, and review status.
- Mark new resources `pending_independent_review` until a second review occurs.

**Gate:** strict schema, checksum, units, table-length, provenance, and exact
value tests, followed by Ruff and the full existing suite.

### Phase 1 — Typed configurations and data contracts

- Implement strict dataclasses and parsers for scene, traffic, system, and CDL
  configurations.
- Implement trajectory, link-condition, large-scale, state, diagnostics, and
  realization contracts.
- Add standard-versus-variation validation and stable configuration hashes.
- Register the two model keys and package resource/schema files.
- Reject unknown fields, illegal mode overrides, nonfinite values, duplicate
  vehicle IDs, inconsistent time axes, invalid lane/street IDs, self-links,
  incompatible metadata, and backend/device/precision mismatches.

**Gate:** contract shape/dtype/device tests, strict configuration tests,
stable hashes, registry tests, and full-suite success.

### Phase 2 — Exact urban-grid geometry

- Generate the rectangular streets, four directed lanes per street,
  intersections, sidewalks, and building-block polygons.
- Assign stable street, lane, intersection, and block IDs.
- Implement lane centerlines and piecewise straight/turn paths.
- Implement robust segment-versus-building tests for link classification.
- Provide backend-neutral snapshot conversion to `TopologyBatch`.
- Add a lightweight scene export for plotting without adding a plotting
  dependency to the core package.

**Gate:** exact dimensions, lane directions, no-overlap invariants,
intersection connectivity, corner/boundary cases, and randomized property
tests across seeds and backends.

### Phase 3 — Vehicle dropping and mobility

- Implement all three standard vehicle dimensions and antenna heights.
- Populate lanes with bumper gaps
  `max(2 m, Exponential(mean = speed * 2 s))`.
- Implement Option A: 100% type 2, 60 km/h, and 0.50/0.25/0.25
  straight/left/right decisions.
- Implement Option B: 20/60/20% types 1/2/3, standard east-west lane speeds,
  stationary north-south lanes, and no intersection direction changes.
- Update positions at the configured mobility interval, 100 ms in standard
  mode, while preserving lane constraints and vehicle orientation.
- Implement `exit` plus explicitly nonstandard `respawn`/periodic policies.

**Gate:** gap distribution, fleet proportions, speed and turn frequencies,
trajectory kinematics, no illegal lane placement, deterministic replay,
vehicle-order independence, and full-suite success.

### Phase 4 — Link state and vehicle blockage

- Classify different-street links as building-obstructed NLOS.
- For same-street links, evaluate the standard
  `min(1, 1.05 exp(-0.0114 d))` LOS probability and complementary NLOSv state.
- Hold LOS/NLOSv state according to the baseline persistence rule.
- Recheck transitions involving building NLOS every mobility update.
- Sample blocker type/height according to the active fleet distribution.
- Implement the three NLOSv height cases and truncated nonnegative blockage
  loss.
- Add an optional geometry-observed blocker diagnostic and an optional
  geometry-driven NLOSv controlled variation; keep the probability baseline
  as the standard mode.

**Gate:** analytic street/building cases, probability and persistence tests,
transition timelines, blocker-height cases, loss moments/quantiles, replay,
and order independence.

### Phase 5 — Path loss and spatially consistent shadowing

- Implement the urban LOS/NLOSv and NLOS path-loss equations.
- Implement the shadowing behavior referenced from TR 36.885, with LOS
  shadowing applied to NLOSv.
- Preserve path loss, shadowing, and vehicle blockage as separate arrays in
  `V2XLargeScaleComponents` and compose them once at CIR voltage level.
- Maintain distance-correlated shadowing across snapshots.
- Define and test state-transition policy for correlated fields rather than
  resampling every 100 ms.

**Gate:** equation reference vectors, monotonicity, shadow standard deviations
and correlation, component composition, disabled-effect behavior, and replay.

### Phase 6 — Three-state stochastic fast fading

- Extract or extend the standard-neutral LSP/ray kernels without changing
  existing TR 38.901 behavior.
- Add distinct urban LOS, NLOSv, and NLOS LSP distributions, correlations,
  cluster counts, ray counts, spreads, XPR, K-factor, and correlation
  distances.
- Follow the referenced TR 38.901 generation procedure only where the Phase-0
  comparison proves equivalence.
- Implement LOS and NLOSv Ricean treatment and NLOS diffuse treatment.
- Add dual transmitter/receiver mobility and the delayed-ray
  `2 * alpha * D` moving-scatterer contribution.
- Preserve clusters and spatial evolution inside a propagation episode.

**Gate:** LSP moments/correlations, tap powers, K-factor, angle distributions,
spatial covariance, temporal autocorrelation, Doppler spectra, link
independence, replay, NumPy/PyTorch parity, and PyTorch gradients.

### Phase 7 — Fixed CDL and antenna integration

- Add exact urban LOS, NLOSv, and NLOS CDL resources.
- Reuse the standard-neutral CDL ray/coefficient kernel after verifying the
  referenced procedure.
- Implement the 6 GHz vehicle antenna Option 1 profiles and placements for
  vehicle types 1–3.
- Apply vehicle yaw to front/rear/rooftop arrays and validate arrival/departure
  conventions.
- Produce loss-scaled, antenna-aware `CIRBatch` realizations and integrate with
  existing time and OFDM transforms/operators.

**Gate:** exact CDL transcription, normalized powers, delays and angles,
antenna boresight/symmetry, explicit MIMO cases, pure-Doppler phase,
OFDM/time application, replay, parity, and gradients.

### Phase 8 — Multi-snapshot orchestration and persistence

- Implement `realize_sequence` over mobility snapshots with immutable state
  carry-over.
- Persist the grid profile, trajectory, IDs, link conditions, channel state,
  large-scale components, CIR snapshots, diagnostics selection, and resource
  provenance with per-file checksums.
- Support cross-backend loading and replay.
- Add filtering by vehicle pair, distance, link state, street, and time.
- Ensure partial link selection produces the same channel as selecting that
  link from the complete run.

**Gate:** intersection-transition end-to-end scenarios, sequence restart,
chunk corruption, cross-backend loading, subset invariance, and full-suite
success.

### Phase 9 — Documentation and release hardening

- Add strict examples for Options A and B plus one controlled variation.
- Add an executable grid-to-trajectory-to-CIR-to-OFDM example.
- Document coordinate/lane conventions, standard versus variation mode,
  state persistence, blockage, Doppler sign, and limitations.
- Add Sphinx API pages and update architecture, capability, and model-selection
  documentation.
- Add pytest markers `v2x`, `tr37885`, `urban_grid`, and `mobility`.
- Verify source and wheel package data in NumPy-only, PyTorch CPU, combined,
  documentation, and optional CUDA jobs.

**Gate:** Ruff, complete pytest suite, strict Sphinx build, package build,
fresh wheel installation, packaged-resource verification, and runnable
examples.

## 9. Test strategy

The implementation is accepted on physical correctness, not merely execution.

### Resource and configuration tests

- exact table values, units, source clauses, checksums, and review state;
- strict unknown-field rejection and standard-mode immutability;
- stable physical hashes excluding backend, device, and seed;
- installed source/wheel resource loading.

### Geometry and traffic tests

- exact grid/lane/intersection dimensions and connectivity;
- legal vehicle placement and bumper separation;
- analytic straight, left-turn, right-turn, and grid-exit trajectories;
- seeded gap, fleet-mixture, and turn-frequency tests with confidence bounds;
- invariant stable IDs and order-independent trajectories.

### Link-state and large-scale tests

- same-street and cross-street analytical cases;
- corner-touching and intersection-boundary policies;
- LOS probability and state persistence;
- NLOSv blocker-height branches and distribution;
- exact path-loss reference values;
- shadow variance and spatial correlation;
- attenuation composition without double counting.

### Fast-fading and antenna tests

- LSP marginal moments and cross-correlations;
- cluster/ray counts, delays, powers, angles, XPR, and K-factor;
- Rayleigh/Ricean distribution checks;
- spatial covariance and MIMO response;
- terminal and moving-scatterer Doppler magnitude/sign;
- temporal autocorrelation and Doppler PSD;
- antenna field reference angles and vehicle-yaw rotation.

### Replay, parity, and integration tests

- exact within-backend replay;
- no process-global RNG access;
- feature/diagnostic/link-evaluation order independence;
- deterministic NumPy/PyTorch agreement;
- PyTorch gradients through continuous geometry, loss, antenna, channel
  transforms, and signal application;
- scene -> mobility -> state -> system/CDL CIR -> OFDM/time output;
- multi-snapshot state transitions and persistence round trips.

Statistical tests use fixed seeds, documented sample counts, confidence-aware
tolerances, and separate slow markers where necessary. They must not use loose
bounds chosen only to make a failing implementation pass.

## 10. Delivery estimate and completion criteria

For one experienced engineer familiar with this repository:

- exact grid plus traffic and link-state simulator: approximately 2–4 weeks;
- large-scale, fast-fading, CDL, antennas, and continuity: approximately
  3–5 additional weeks;
- persistence, documentation, packaging, and statistical calibration:
  approximately 1–2 additional weeks.

The total expected effort is approximately **6–11 engineer-weeks**. Independent
table review and external calibration may add calendar time without requiring
continuous implementation effort.

The increment is complete only when:

1. Standard mode reproduces every included V15.3.0 urban-grid assumption.
2. Controlled variations are explicitly identified in configuration and
   provenance.
3. Seeded scenes, link states, CIRs, and transformed outputs replay exactly
   within each backend.
4. Analytical, statistical, syntax/shape/device, integration, and persistence
   tests all pass.
5. The complete pre-existing suite remains green after every phase.
6. Ruff, strict Sphinx, source distribution, wheel installation, and packaged
   resource checks pass.
7. Resources remain labeled `pending_independent_review` until a second review
   is recorded; the documentation does not claim formal conformance before
   that review.
