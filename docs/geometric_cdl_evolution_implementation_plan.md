# Geometrically Consistent CDL Cluster-Evolution Engine

## Summary

Implement a backend-neutral `geometric_cdl_evolution` model that generates
everything upstream of the existing time-varying CDL coefficient engine:

```text
trajectory + scenario/profile
        ↓
spatially correlated random fields
        ↓
LOS state + large-scale parameters
        ↓
segment-anchor cluster/ray initialization
        ↓
FBS/LBS scatterer geometry
        ↓
geometric drifting along the trajectory
        ↓
segment birth/death and transition weights
        ↓
EvolvingRayBatch
        ↓
existing CDL coefficient/CIR engine
```

The implementation will provide:

- PyTorch-first and native NumPy execution.
- Dual mobility and batched multi-link support.
- Correlated multiband realizations.
- CDL-A through CDL-E and all 3GPP TR 38.901 scenario families: RMa, UMa,
  UMi, InH, and InF.
- Algorithmic and statistical parity with documented QuaDRiGa behavior, not
  identical MATLAB random samples.
- Gradients through deterministic geometric drifting, but not through
  stochastic sampling, scatterer fallback decisions, cluster matching, or
  birth/death assignment.
- An independent Apache-2.0 implementation based on standards and documented
  algorithms; QuaDRiGa is used only as an optional external oracle.

This extends the existing [wireless implementation plan](wireless_implementation_plan.md):
Phase 14's stateful-mobility deferral is replaced with the stateful evolution
work below. Coefficient generation, antenna response, waveform application,
OFDM transforms, and noise remain outside this plan.

## Public interfaces and data contracts

### New request and trajectory types

Add `TopologyTrajectoryBatch` with backend arrays:

- `times_s`: `[batch, time]`, strictly increasing.
- Tx position, velocity, orientation: `[batch, time, tx, 3]`.
- Rx position, velocity, orientation: `[batch, time, rx, 3]`.
- `link_mask`: optional `[batch, time, rx, tx]`.
- Endpoint identity arrays that remain stable over time.

Add `ScenarioTimelineBatch`:

- `scenario_index`: `[batch, time, rx, tx]`.
- Immutable mapping from integer indices to typed scenario/profile identifiers.
- Optional fixed LOS mask; otherwise LOS state is sampled spatially.
- A scenario or LOS-state change always forces a segment boundary.

Add `ClusterEvolutionRequest` containing:

- Trajectory and scenario timeline.
- Carrier frequencies `[frequency]`.
- Execution context and injected `RandomStream`.
- Optional explicit segment boundaries.
- Request identifier and provenance metadata.

The request validates backend/device consistency and verifies that the input
sampling grid is sufficiently dense for the highest carrier. For dual
mobility, the default minimum is two samples per half wavelength. Resampling
remains a separate helper so generation never silently changes the requested
timeline.

### Realization and state types

Implement `GeometricCDLModel` as:

```python
ChannelModel[ClusterEvolutionRequest, GeometricCDLRealization]
```

Expose lower-level operations for testing and reuse:

```python
initialize_anchors(request) -> GeometricClusterState
evolve(state, trajectory) -> EvolvingRayBatch
realize(request) -> GeometricCDLRealization
```

`GeometricClusterState` stores padded backend arrays and masks, never per-ray
Python objects:

- Segment anchors and validity.
- SOS field definitions and phases.
- Anchor LSP values.
- Cluster/ray identities and type codes: LOS, ground reflection, or NLOS.
- FBS/LBS positions:
  `[batch, rx, tx, segment, frequency, cluster, ray, xyz]`.
- Initial phases, XPR parameters, ray coupling, base powers, and masks.
- Geometry-solver status and fallback flags.
- Resolved configuration and RNG lineage.

`EvolvingRayBatch` is the handoff to the existing coefficient engine:

- Cluster powers: `[batch, rx, tx, frequency, cluster, time]`.
- Per-ray power fractions and amplitude transition weights:
  `[batch, rx, tx, frequency, cluster, ray, time]`.
- Per-ray delays or path lengths using the same axes.
- Time-varying AoD, AoA, ZoD, and ZoA.
- XPR and initial phase.
- Cluster/ray masks, type codes, segment IDs, and transition-generation IDs.
- Optional FBS/LBS diagnostics without changing random draws.

Transition overlaps retain old and new cluster generations simultaneously. The
coefficient engine sums both contributions. This avoids interpolating
incompatible scatterer geometries.

Extend the existing `RayBatch` with optional static `ray_delay_offsets_s` and
`ray_power_weights_linear`. Selecting one frequency and time from
`EvolvingRayBatch` must yield a compatible static `RayBatch`.

### Configuration and resources

Create a frozen `GeometricCDLEvolutionConfig` with nested typed sections:

- Standard/profile version and scenario resource references.
- Mobility mode: `hybrid`, `drift_only`, or `spatial_consistency_only`; default
  `hybrid`.
- Spatial field ACF, sinusoid count, and decorrelation controls.
- Explicit or automatic segmentation.
- Cluster/subpath configuration.
- Two-bounce solver and fallback policy.
- Ground-reflection controls.
- Absolute/relative delay convention.
- Multiband correlation policy.
- Diagnostic and numerical tolerances.

Defaults:

- Combined Gaussian/exponential ACF with 300 sinusoids.
- Automatic segments using profile stationarity distance.
- Overlap fraction `0.5`.
- Two-bounce geometry with deterministic single-bounce fallback.
- Absolute propagation delays.
- Ground reflection disabled unless selected by the profile.
- `legacy` and `Laplacian` subpath methods supported; `mmMAGIC` deferred.
- No 3GPP baseline planar-Doppler mode.

Store scientific parameters in versioned YAML/JSON resources:

- CDL-A through CDL-E tables.
- RMa, UMa, UMi, InH, and InF LOS/NLOS/O2I variants.
- Eight LSP means, standard deviations, dependencies, decorrelation distances,
  and correlation matrices.
- Cluster counts, per-cluster shadowing, delay/angle spread controls, XPR, and
  ground parameters.
- Specification version, table/section, units, applicability range,
  transcription review, and checksum.

Use TR 38.901 v16.1.0 as the QuaDRiGa comparison baseline. Add v19.2 resources
as a separate version namespace without changing engine code.

## Algorithms

### Spatial field bank

Implement an immutable `SpatialFieldBank` using sum-of-sinusoids fields:

:::{math}
z(\mathbf{x})=a\sum_l \cos(2\pi\mathbf{k}_l^\mathsf{T}\mathbf{x}+\phi_l).
:::

Generate the sinusoid basis independently rather than copying QuaDRiGa's
precomputed coefficients:

- Add an offline deterministic tool that fits the selected ACF.
- Commit generated bases as versioned resources with optimizer settings and
  checksums.
- Randomize only field phases at realization time.
- Evaluate fields in position chunks to control memory.

Provide normal and uniform marginal transforms. For dual mobility, evaluate
both endpoints and normalize their sum using the ACF at the endpoint
separation. Enforce reciprocity by phase sharing:

- DS, KF, SF, XPR, and delay fields are invariant under endpoint exchange.
- ASD/ASA fields exchange roles.
- ZSD/ZSA fields exchange roles.

Use semantic RNG stream names per scenario and field. Field generation must be
invariant to batching, diagnostic flags, and iteration order.

### LOS and LSP evolution

For each active scenario:

1. Evaluate a correlated uniform field for LOS state unless fixed by the
   request.
2. Evaluate eight independent normal SOS fields for DS, KF, SF, ASD, ASA, ZSD,
   ZSA, and XPR.
3. Apply the scenario's positive-semidefinite {math}`8\times8` correlation matrix
   using `matrix_sqrt_psd`.
4. Apply frequency-, distance-, height-, and elevation-dependent means and
   standard deviations.
5. Transform logarithmic values to SI/linear domains.
6. Evaluate and retain the complete LSP trajectory for diagnostics.

At a segment anchor, all LSPs initialize the new cluster set. Inside a
geometric-drifting segment:

- SF and KF update continuously and rescale total and deterministic/diffuse
  powers.
- DS and angular-spread fields are reported but do not continuously
  re-normalize fixed scatterer geometry.
- DS and angular spreads affect the next segment initialization.

This preserves local geometric consistency while allowing long-range
statistical evolution.

### Cluster and ray initialization

For each segment anchor and frequency:

1. Reserve cluster 0 for LOS and, when enabled, cluster 1 for ground
   reflection.
2. Draw spatially correlated NLOS delays as {math}`-\log U`.
3. Draw four correlated angular variables without sorting or identity-changing
   strongest-path operations.
4. Generate unnormalized cluster powers from delay and angular offsets,
   including per-cluster lognormal variation.
5. Normalize NLOS power and apply KF; divide deterministic power between LOS
   and ground reflection.
6. Analytically rescale NLOS delays to match target RMS DS.
7. Iteratively rescale angular clouds to target ASD/ASA/ZSD/ZSA, with pole
   reflection and Cartesian rotation to the physical LOS direction.
8. Apply absolute NLOS TOA offset when configured.
9. Expand clusters into subpaths using `legacy` or `Laplacian` offsets.
10. Use independent, explicitly named permutations for AoD, AoA, ZoD, and ZoA
    coupling.
11. When spatial consistency and per-cluster delay spread are enabled, split
    every eligible cluster rather than selecting the instantaneous strongest
    two.
12. Generate spatially correlated XPR, polarization parameters, and initial
    phases.

Across frequencies, reuse the same field identities and random ranks while
applying frequency-specific LSPs, powers, XPR, and spread scaling.
Frequency-specific scatterer geometry is allowed where required to satisfy
different per-band delay or angular spreads.

### Scatterer geometry

For each ray, construct departure and arrival unit vectors and solve:

:::{math}
\min_{a,b\ge0}\|\mathbf{x}_{Tx}+a\mathbf{u}_D-
(\mathbf{x}_{Rx}+b\mathbf{u}_A)\|^2
:::

subject to the requested total path length:

:::{math}
a+b+\|\mathbf{x}_{Tx}+a\mathbf{u}_D-
(\mathbf{x}_{Rx}+b\mathbf{u}_A)\|
=d_{\mathrm{LOS}}+c\tau.
:::

Implement a backend-native, batched projected Gauss-Newton solver with:

- Precision-dependent convergence tolerance.
- Fixed maximum iteration count.
- Residual and validity diagnostics.
- No SciPy dependency or CPU round-trip.

Use direct geometry for LOS and image geometry for ground reflection. If the
constrained two-bounce solution is infeasible, apply the configured policy:

- Default: deterministic single-bounce solution with `FBS == LBS`.
- Optional strict mode: raise an error.
- Optional masking mode: invalidate the ray.

Fallbacks are recorded in realization metadata and tested statistically by
profile.

### Drifting

Within each segment, FBS/LBS positions remain fixed. For every trajectory
sample:

1. Rotate endpoint array frames using the supplied orientations.
2. Recompute Tx-to-FBS and LBS-to-Rx vectors.
3. Derive time-varying departure and arrival angles.
4. Compute full ray length and absolute or LOS-relative delay.
5. Return path length so the downstream engine derives phase and Doppler
   geometrically.
6. Apply continuously evaluated SF/KF scaling.
7. Preserve masks and ray identity.

The drift kernel contains no Python loop over links, clusters, rays, or time.
Group-level loops over scenario, segment, and bounded solver iterations are
permitted.

PyTorch gradients must pass from positions/orientations through vector
geometry, angles, lengths, and delays. The sampled state and matching decisions
are treated as constants.

### Segmentation and transitions

Automatic segmentation creates a new segment when:

- Scenario or sampled LOS state changes; or
- The maximum Tx/Rx displacement from the anchor exceeds the profile
  stationarity distance.

Explicit segment boundaries override automatic distance segmentation but not
scenario validity checks.

Adjacent segment realizations are separate cluster sets sampled from the same
spatial field bank. Cluster identity is not claimed across segments.

For transition overlaps:

1. Exclude deterministic LOS/GR components from NLOS matching.
2. Greedily choose old/new cluster fade order to minimize delay-spread MSE plus
   maximum total-power deviation, matching QuaDRiGa's documented cost behavior.
3. Divide the overlap into as many subintervals as the smaller NLOS cluster
   count.
4. Use

   :::{math}
   r(w)=\tfrac12[1+\sin(\pi(w-\tfrac12))]
   :::

   as the power ramp.
5. Scale NLOS amplitudes by {math}`\sqrt{1-r}` and {math}`\sqrt r`.
6. Scale old/new LOS and GR coefficients by {math}`1-r` and {math}`r`; represent these as
   separate contributions with explicit amplitude weights.
7. Fade unmatched NLOS clusters across the full overlap.
8. Maintain a fixed padded capacity of two cluster generations during overlaps
   and use masks outside them.

Matching and slot assignment are nondifferentiable. Once assignments are
fixed, geometric evaluation and ramp functions remain differentiable.

## Implementation phases

### Phase 1 - Reconcile the existing roadmap and contracts

- Update the existing roadmap so stateful evolution is inserted after
  cluster/ray generation and before coefficient assembly.
- Remove duplicate CDL motion logic from the old Phase 9 scope.
- Document that path loss remains an independent final transform.
- Finalize axes, angle conventions, delay conventions, and coefficient-engine
  handoff.
- Add empty typed contracts and shape-validation tests.

Gate: synthetic trajectory/state/evolving-ray objects validate for both
backends and can be accepted by a stub coefficient adapter.

### Phase 2 - Backend primitives and batched state

Extend `ArrayBackend` only with operations required by the algorithms:

- `where`, clipping, sign, floor, and remainder.
- `asin`, `acos`, and tangent.
- `argsort`, `argmin`, and gather/take-along-axis.
- Batched linear solve where required.
- Boolean reductions and masked replacement.

Extend `RandomStream` with batched permutations or implement them through
uniform random keys plus stable `argsort`. Preserve semantic child-stream
behavior.

Implement the new trajectory, scenario, state, LSP-trajectory, and evolving-ray
containers.

Gate: protocol conformance, shape/mask validation, device/dtype preservation,
and deterministic stream-isolation tests pass.

### Phase 3 - Configuration and standard resources

- Add strict schemas and frozen configuration parsers.
- Transcribe CDL-A-E and all scenario resources from the selected standards.
- Add applicability and PSD validation.
- Generate independent SOS basis resources and checksums.
- Register `wireless/geometric_cdl_evolution`.

Gate: every resource round-trips, rejects unknown fields, passes
checksum/table-shape tests, and records exact provenance.

### Phase 4 - Spatial fields and LSP engine

- Implement SOS evaluation, marginal transforms, dual-mobility combination,
  and reciprocity mappings.
- Implement LOS-state sampling.
- Implement LSP cross-correlation and scenario-dependent transformations.
- Add multiband shared-field evaluation.

Gate: empirical ACF, decorrelation distance, LSP moments, cross-correlation,
reciprocity, and multiband correlation meet declared statistical tolerances.

### Phase 5 - Static cluster/ray initialization

- Implement delays, powers, K-factor split, exact DS scaling, angular scaling,
  subpaths, coupling, XPR, phases, masks, and optional ground reflection.
- Ensure spatial-consistency mode never sorts cluster identities.
- Produce a static `RayBatch` before adding geometry.

Gate: all scenario/profile invariants pass for NumPy and PyTorch; CDL-A-E
retain their prescribed reference values before geometric adaptation.

### Phase 6 - FBS/LBS geometry

- Implement the two-bounce solver, single-bounce fallback, LOS, and ground
  reflection.
- Add solver diagnostics and failure policies.
- Validate reconstructed delay and direction against input targets.

Gate: feasible fixtures reconstruct path length and angles within
precision-specific tolerance; infeasible fixtures take the declared fallback
path deterministically.

### Phase 7 - Drifting and gradient support

- Implement vectorized drift evaluation for static Tx, static Rx, dual
  mobility, rotations, curved trajectories, and multiple carriers.
- Add absolute/relative delay handling and dynamic SF/KF scaling.
- Add the static-`RayBatch` adapter.

Gate: stationary links remain constant; geometric phase derivative matches
analytic Doppler; reciprocal links exchange arrival/departure directions;
PyTorch gradient checks pass away from wraps and fallbacks.

### Phase 8 - Segment transitions and birth/death

- Implement segmentation, segment-anchor generation, greedy transition
  ordering, two-generation overlap storage, and amplitude ramps.
- Preserve random-stream isolation when segment count, diagnostics, or output
  selection changes.

Gate: no coefficient discontinuity occurs after downstream summation;
transition endpoints reproduce their source segments; total power and delay
spread remain within configured tolerances.

### Phase 9 - Full orchestration, scenarios, and multiband

- Assemble the top-level model.
- Complete RMa, UMa, UMi, InH, and InF variants.
- Validate batched multi-link, dual-mobility, scenario changes, LOS changes,
  and correlated multiband operation.
- Add serialization/reload of field banks and geometric state.
- Add chunking for large trajectories without changing results.

Gate: chunked and unchunked executions produce the same realization;
serialized state resumes exactly; all supported profiles feed the existing
coefficient engine without resampling.

### Phase 10 - Reference validation and optimization

- Add optional MATLAB/Octave reference tooling that invokes a separately
  installed QuaDRiGa checkout.
- Compare documented statistics and physical invariants, not seeded sample
  identity.
- Never import, vendor, or translate QuaDRiGa source into production code.
- Benchmark PyTorch CPU/CUDA and NumPy.
- Optimize only after parity tests exist; retain a clear reference kernel for
  complicated geometry and transition operations.

Gate: all correctness suites pass, reference metrics meet documented
tolerances, and no production dependency on QuaDRiGa, MATLAB, or SciPy exists.

## Test plan

Required suites include:

- Configuration/resource integrity for every profile and standard version.
- SOS ACF shape, decorrelation distance, marginals, reciprocity, and replay.
- LOS-state correlation and fixed-state override.
- LSP means, standard deviations, cross-correlation, distance correlation, and
  multiband correlation.
- Exact or tolerance-bounded DS and angular-spread matching.
- Cluster normalization, LOS/NLOS/GR power split, XPR, phases, coupling, and
  masks.
- No cluster reordering under spatial consistency.
- Two-bounce reconstruction and deterministic fallback.
- Static, linear, curved, rotated-array, and dual-mobility drifting.
- Analytic Doppler from path-length derivatives.
- Birth/death endpoints, ramp shape, transition continuity, total power, and DS
  cost.
- NumPy/PyTorch deterministic parity for fixed inputs.
- Statistical equivalence rather than random-sample equality.
- CPU/CUDA and single/double precision.
- PyTorch gradient tests for deterministic drift.
- Random-stream isolation when diagnostics, chunk size, optional GR, or output
  selection changes.
- Serialization/replay with configuration digest and field-state provenance.
- End-to-end CDL-A-E and every RMa/UMa/UMi/InH/InF LOS/NLOS variant.
- Optional QuaDRiGa oracle comparisons for ACF, LSP statistics,
  path-delay/angle evolution, Doppler bounds, and transition metrics.

## Assumptions and exclusions

- The existing time-varying CDL coefficient engine accepts time-varying ray
  parameters and per-ray amplitude weights; only a thin adapter may be needed.
- Path loss, antenna field evaluation, polarization-to-coefficient conversion,
  CIR assembly, and waveform application are existing/downstream
  responsibilities.
- Standardized CDL tables seed cluster parameters, but the packaged QuaDRiGa
  baseline CDL implementation is not reproduced.
- Numerical MATLAB replay parity is not required.
- Temporal blockage evolution, environment-aware deterministic ray tracing,
  `mmMAGIC` subpaths, and differentiable birth/death relaxations are excluded.
- Ground reflection is supported but opt-in.
- Segment-local scatterers are persistent; across segments the implementation
  promises spatial correlation and smooth replacement, not globally persistent
  physical cluster identity.
- All stochastic operations use injected, semantically forked random streams,
  and all persistent outputs carry schema version, resolved configuration, seed
  lineage, standard reference, software version, and fallback diagnostics.
