# Wireless Channel Implementation Plan

## 1. Purpose

This document is the implementation roadmap for the wireless portion of the
Channel Dataset project. The goal is to provide reusable physical-layer channel
models with:

- PyTorch as the default and first-priority numerical backend;
- full NumPy support for CPU-only use, reference calculations, and environments
  where PyTorch is unavailable;
- model parameters and standard tables stored outside source code;
- repeatable random generation without process-global random state;
- permanent unit and integration tests for every implemented component;
- a stable path from simple fading models to 3GPP TR 38.901 system-level
  channels.

This is a functional, standards-first reimplementation. Sionna may be used as
an optional behavioral oracle during development, but its source code, comments,
internal module layout, tests, and generated source fragments must not be copied
or mechanically translated into this repository.

Because no Sionna source is to be copied, new files should carry this project's
normal authorship and license notices, not upstream source headers. If code or
data is ever copied or adapted from an external project in the future, its
license and attribution requirements must be handled separately. Calling a use
case “research” does not by itself waive those requirements.

This approach is an independent reimplementation, not a formal legal
clean-room process unless specification review and implementation are performed
by separate people under an explicitly documented clean-room procedure.

## 2. Scope

### 2.1 Included

The first wireless implementation program covers:

1. Backend-neutral array and random-number interfaces.
2. Batched channel impulse response (CIR), cluster, ray, topology, and antenna
   representations.
3. AWGN, Rayleigh block fading, flat MIMO fading, and spatial correlation.
4. Time-domain and OFDM channel-response generation and application.
5. 3GPP tapped delay line (TDL) models.
6. Antenna elements, panels, arrays, orientation, and polarization.
7. 3GPP clustered delay line (CDL) models.
8. 3GPP RMa, UMa, UMi, InH, and InF system-level scenarios.
9. Large-scale parameters, clusters, rays, path loss, shadowing, blockage, and
   spatial consistency.
10. Dataset serialization, replay, and channel metrics.

### 2.2 Deferred

The following are deliberately outside this program:

- binary symmetric, erasure, Z, and other discrete information-theory channels;
- fiber and free-space optical models;
- transmitter, receiver, coding, and link-layer implementations;
- plotting inside model classes;
- ray tracing tied to a particular scene engine;
- full stateful mobility and birth/death evolution until the static and
  snapshot-consistent models are complete.

## 3. Non-negotiable design rules

1. **PyTorch is the default backend.** Calling code that does not select a
   backend receives PyTorch tensors. PyTorch implementations are written and
   optimized first.
2. **NumPy is a supported backend, not a conversion wrapper.** NumPy calls must
   execute with NumPy arrays and NumPy random generators without importing
   PyTorch.
3. **The public physics API is backend-neutral.** Backend-specific tensor types
   may appear in returned array fields, but model concepts, configuration, axis
   order, and units must not change by backend.
4. **Do not implement a lowest-common-denominator tensor framework.** Use a
   small backend protocol for shared operations, and permit backend-specific
   optimized kernels behind the same public contract.
5. **No global RNG.** Every stochastic operation receives a `RandomStream` or a
   child stream derived from it.
6. **No scientifically meaningful parameters hidden in code.** Profile values,
   standard tables, default delay spreads, scenario constants, and model
   selections belong in versioned YAML or JSON files.
7. **Execution settings are not physical model settings.** Device, precision,
   backend, batching, and seed must be kept separate from propagation-model
   parameters.
8. **Generation, representation conversion, signal application, and noise are
   separate operations.** Each must be usable and testable independently.
9. **SI units are used internally.** Field names carry suffixes such as `_hz`,
   `_s`, `_m`, `_mps`, `_rad`, `_db`, and `_linear`.
10. **Tensor axes are explicit and validated.** Public tensor containers must
    document their canonical axis order and reject incompatible shapes.
11. **Every implementation change includes tests.** A component is not complete
    until its permanent test file is committed and passes on every supported
    backend relevant to that component.
12. **Standards support is versioned.** A profile must identify the exact
    specification family, version, table or section, and known applicability
    limits.

## 4. Target package structure

The target structure is:

```text
src/channel_dataset/
  backends/
    base.py                    Backend and RandomStream protocols
    registry.py                Backend discovery and default selection
    numpy_backend.py           NumPy implementation
    torch_backend.py           PyTorch implementation, lazily imported
  core/
    axes.py                    Canonical axis names and shape validation
    context.py                 Semantic simulation context
    models.py                  Backend-neutral model interfaces
    paths.py                   Scalar semantic paths and clusters
    realization.py             Realization metadata and provenance
  channels/wireless/
    types.py                   CIRBatch and wireless batch value objects
    requests.py                Sampling and response requests
    math/
      conversions.py           dB/linear and unit conversions
      angles.py                Angle wrapping and coordinate helpers
      random.py                Distribution helpers
      correlation.py           Correlation matrix utilities
    noise/
      awgn.py
    fading/
      rayleigh.py
      flat.py
      spatial_correlation.py
    transforms/
      time_lags.py
      cir_to_time.py
      subcarriers.py
      cir_to_ofdm.py
    operators/
      apply_flat.py
      apply_time.py
      apply_ofdm.py
    datasets/
      cir_dataset.py
      serialization.py
    tr38901/
      config.py
      tdl.py
      antennas.py
      topology.py
      cdl.py
      large_scale.py
      rays.py
      coefficients.py
      blockage.py
      spatial_consistency.py
      metrics.py
      scenarios/
        base.py
        rma.py
        uma.py
        umi.py
        inh.py
        inf.py

configs/wireless/
  schemas/
  profiles/
  3gpp/
    tr_38_901/
      v16_1/
      v19_2/
    ts_38_101_4/
      v19_2_2/

tests/
  unit/backends/
  unit/wireless/
  unit/wireless/tr38901/
  integration/wireless/
  statistical/wireless/
  reference/wireless/
```

Avoid generic dumping-ground modules such as a single multi-thousand-line
`utils.py`. A helper should live with the concept that owns it.

## 5. Public data contracts

### 5.1 Execution context

Create an immutable `ExecutionContext` with:

- `backend`: `"torch"` or `"numpy"`, defaulting to `"torch"`;
- `precision`: `"single"` or `"double"`;
- `device`: meaningful for PyTorch and restricted to CPU for NumPy;
- optional compilation/performance flags that do not affect the physical model.

The execution context constructs or retrieves a backend instance. It does not
own physical parameters or a global seed.

### 5.2 Random streams

Define a `RandomStream` protocol supporting at least:

- `uniform(shape, low, high, dtype)`;
- `normal(shape, mean, stddev, dtype)`;
- `complex_normal(shape, variance, dtype)`;
- `integers(shape, low, high)`;
- `bernoulli(shape, probability)`;
- `permutation(n)`;
- `fork(label)` for deterministic component-specific child streams.

Implement `TorchRandomStream` with `torch.Generator` and `NumpyRandomStream`
with `numpy.random.Generator`. A parent seed plus a stable child label must
produce repeatable child streams. Tests must not assume that NumPy and PyTorch
produce identical sample values; they must verify reproducibility and
distributional equivalence within each backend.

### 5.3 Sampling request

Create an immutable `ChannelSamplingRequest` containing:

- batch size;
- number of time samples;
- sampling frequency in hertz;
- optional topology batch;
- execution context;
- random stream;
- optional request identifier for provenance.

Model configuration remains on the model object. Values that change per call
belong in the request.

### 5.4 Batched CIR

Create a generic `CIRBatch` with canonical fields:

- `coefficients`: complex array with axes
  `[batch, rx, rx_antenna, tx, tx_antenna, path, time]`;
- `delays_s`: real array with axes `[batch, rx, tx, path]` or, only where
  explicitly required, `[batch, rx, tx, path, time]`;
- `path_mask`: optional boolean array for batches with different path counts;
- `metadata`: model identity, schema version, standard version, units, seed or
  stream identity, and resolved configuration digest.

Validation must check rank, shared dimensions, real/complex dtype, nonnegative
delays, finite values where required, and backend consistency.

### 5.5 Cluster and ray batches

Create `ClusterBatch` and `RayBatch` rather than instantiating a Python object
for every simulated ray. At minimum, `RayBatch` contains:

- cluster delays and normalized powers;
- azimuth and zenith angles of arrival and departure in radians;
- cross-polarization ratios;
- initial phases;
- optional blockage loss;
- cluster/ray validity masks;
- sort and strongest-cluster indices where required by the standard.

Provide an explicit adapter such as `select_paths(batch, rx, tx)` that converts
one selected link into semantic `PropagationPath` and `PathCluster` objects for
inspection or serialization. Never perform this conversion implicitly in the
hot simulation path.

### 5.6 Response containers

Define distinct containers:

- `TimeChannelResponse`, with axes
  `[batch, rx, rx_antenna, tx, tx_antenna, output_time, lag]`;
- `OFDMChannelResponse`, with axes
  `[batch, rx, rx_antenna, tx, tx_antenna, symbol, subcarrier]`;
- `FlatChannelResponse`, with axes
  `[batch, rx_antenna, tx_antenna]` or a documented multi-link extension.

These containers validate shape and backend exactly as `CIRBatch` does.

## 6. Configuration system

### 6.1 Separate three kinds of configuration

1. **Execution configuration**: backend, device, precision, batch controls, and
   seed.
2. **Model configuration**: user-selected physical settings such as carrier
   frequency, speed, delay spread, antenna choice, and direction.
3. **Standard parameter resources**: versioned tables of powers, delays,
   correlations, scenario coefficients, and applicability limits.

Do not merge these into one untyped dictionary.

### 6.2 Configuration envelope

Extend the common configuration envelope so it can identify:

- schema version;
- model family and name;
- model profile or standard resource reference;
- model parameters;
- execution parameters;
- optional metadata such as experiment label.

All configuration documents must reject unknown fields by default.

### 6.3 Typed validation

For each model:

1. Define a frozen typed configuration class.
2. Parse the external document into that class.
3. Validate units, ranges, required combinations, matrix shapes, positive
   semidefiniteness where appropriate, and standard applicability constraints.
4. Resolve profile inheritance before model construction.
5. Store the fully resolved configuration in realization provenance.

The model constructor receives the typed object, never the raw mapping.

### 6.4 Standard resources

Create parameter resources directly from the applicable published standard,
not by copying Sionna JSON files. Each resource must include adjacent metadata
with:

- specification identifier and version;
- table, annex, or section reference;
- units for every field;
- transcription date;
- independent reviewer/check status;
- known frequency, bandwidth, distance, or scenario restrictions.

Add schema tests and table-level checksum tests so accidental edits are caught.

## 7. Testing policy

### 7.1 Test categories

Use these permanent categories:

- `tests/unit`: fast deterministic tests for one class or function;
- `tests/integration`: multiple implemented components working together;
- `tests/statistical`: seeded distribution and correlation tests;
- `tests/reference`: optional comparisons with external implementations or
  published reference data;
- `tests/performance`: opt-in benchmarks, not part of ordinary correctness
  runs.

### 7.2 Backend parameterization

Every backend-neutral component test must be parameterized over `torch` and
`numpy`. Provide fixtures named `backend_name`, `backend`, `random_stream`, and
`execution_context`.

PyTorch-specific tests additionally cover:

- CPU device behavior;
- CUDA when available;
- `complex64`/`float32` and `complex128`/`float64`;
- gradient preservation for deterministic signal-application operations;
- compilation compatibility only after the eager implementation is correct.

NumPy-specific tests cover:

- operation without PyTorch installed or imported;
- `complex64`/`float32` and `complex128`/`float64`;
- CPU-only error handling for unsupported device requests.

### 7.3 Required tests for every component

Every new public class or function must have tests for:

1. Nominal behavior.
2. Shape and axis contract.
3. Dtype and backend preservation.
4. Boundary values.
5. Invalid inputs and clear exceptions.
6. Deterministic reproducibility when stochastic.
7. At least one analytic or physical sanity check.
8. NumPy/PyTorch numerical parity for deterministic calculations.
9. Serialization or configuration round-trip when applicable.

### 7.4 Statistical testing rules

- Fix seeds and sample counts.
- Test moments, quantiles, correlations, or empirical CDF distance instead of
  exact random samples.
- Set tolerances from expected sampling error, not from observed failures.
- Separate slow high-confidence tests from fast smoke-level statistical tests.
- Avoid flaky one-sided thresholds near the expected value.
- Record formulas and expected distributions in test docstrings.

### 7.5 External reference comparisons

If Sionna is used as an oracle:

- invoke an independently installed copy only from optional reference tooling;
- never import it from production code;
- never vendor its test code or source;
- compare public outputs and statistical behavior;
- record the oracle package name, version, configuration, and seed in generated
  result metadata for scientific reproducibility;
- do not treat agreement with one implementation as proof of standards
  conformance.

## 8. Detailed implementation phases

Each phase below has ordered instructions and a required test deliverable. Do
not begin a dependent phase until the prior phase's acceptance gate passes.

### Phase 0 — Project and dependency preparation

#### Implementation

1. Select and add this project's own license.
2. Add optional dependency groups:
   - `wireless-torch` for PyTorch;
   - `wireless-numpy` for NumPy and required scientific helpers;
   - `wireless` installing both, with PyTorch as the documented default;
   - `test` for pytest and test utilities.
3. Keep NumPy available to the core project but ensure importing NumPy channel
   code does not import PyTorch.
4. Add pytest markers: `torch`, `numpy`, `cuda`, `statistical`, `slow`,
   `reference`, and `performance`.
5. Add continuous-integration jobs for:
   - PyTorch CPU, the primary required job;
   - NumPy-only installation;
   - combined installation;
   - optional CUDA validation where infrastructure permits.
6. Document supported Python, NumPy, and PyTorch version ranges.

#### Tests

Create `tests/unit/test_package_imports.py` and verify:

- the base package imports with core dependencies;
- the NumPy backend imports without importing PyTorch;
- the PyTorch backend reports an actionable optional-dependency error when
  unavailable;
- default backend resolution selects PyTorch when installed.

#### Acceptance gate

All installation variants import successfully and test discovery works without
collecting unavailable-backend failures.

### Phase 1 — Backend and random-stream foundation

#### Implementation

1. Define `ArrayBackend` and `RandomStream` protocols.
2. Define a backend registry with lazy imports.
3. Implement the PyTorch backend first:
   - dtype resolution;
   - device validation;
   - array creation and conversion;
   - elementwise math;
   - reductions;
   - broadcasting and shape operations;
   - matrix multiplication/einsum;
   - FFT operations;
   - Cholesky/eigendecomposition helpers;
   - explicit `torch.Generator` random operations.
4. Implement the NumPy backend against the same protocol.
5. Add stable child-stream derivation. Document whether a child stream is
   derived by seed hashing or `SeedSequence`; ensure Python's randomized
   process hash is never used.
6. Add helpers to assert that all arrays participating in an operation belong
   to the same backend and device.
7. Keep backend-specific fast paths private.

#### Tests

Create:

- `tests/unit/backends/test_registry.py`;
- `tests/unit/backends/test_torch_backend.py`;
- `tests/unit/backends/test_numpy_backend.py`;
- `tests/unit/backends/test_random_streams.py`;
- `tests/unit/backends/test_backend_parity.py`.

Test every protocol operation, dtype mapping, invalid device, seeded replay,
child-stream stability, child-stream independence, and deterministic
cross-backend mathematical parity.

#### Acceptance gate

Both backends satisfy a shared protocol conformance suite. No stochastic code
outside the backend layer accesses global NumPy or PyTorch RNG state.

### Phase 2 — Axes and batched wireless types

#### Implementation

1. Define canonical axis enums/constants.
2. Implement reusable rank and shared-dimension validators.
3. Implement `CIRBatch`, response containers, `TopologyBatch`,
   `LargeScaleParameters`, `ClusterBatch`, and `RayBatch` incrementally.
4. Make containers immutable at the Python-object level while allowing tensor
   data to remain tensor data.
5. Add selection methods that return views where possible.
6. Add scalar path/cluster conversion for a selected link.
7. Define realization provenance and configuration-digest fields.
8. Update the common `ChannelModel` interface so it is no longer typed to
   NumPy arrays or `numpy.random.Generator`.

#### Tests

Create one test file per container under `tests/unit/wireless/`, including:

- `test_cir_batch.py`;
- `test_channel_responses.py`;
- `test_topology_batch.py`;
- `test_large_scale_parameters.py`;
- `test_cluster_batch.py`;
- `test_ray_batch.py`;
- `test_path_conversion.py`.

Test valid and invalid ranks, mismatched dimensions, masks, nonnegative delays,
backend mismatch, selection semantics, and scalar conversion.

#### Acceptance gate

A synthetic multi-BS, multi-UT, multi-antenna CIR can be represented and
selected without copying entire arrays or generating per-ray Python objects.

### Phase 3 — Configuration and resource loading

#### Implementation

1. Extend the configuration envelope without breaking schema-version handling.
2. Add separate typed execution and model configurations.
3. Implement profile references and deterministic override resolution.
4. Add resource lookup through package resources rather than current-working-
   directory assumptions.
5. Add JSON schemas for the common envelope and each initial wireless model.
6. Add resolved-configuration hashing with stable key ordering.
7. Make unknown fields errors.
8. Make unit-bearing fields explicit; reject ambiguous aliases.

#### Tests

Create:

- `tests/unit/wireless/test_config_envelope.py`;
- `tests/unit/wireless/test_profile_resolution.py`;
- `tests/unit/wireless/test_config_hashing.py`;
- `tests/unit/wireless/test_resource_loading.py`.

For every model added later, create a corresponding `test_<model>_config.py`.
Test YAML and JSON equivalence, missing fields, unknown fields, invalid units,
profile overrides, package installation resource lookup, and stable hashes.

#### Acceptance gate

A model can be constructed entirely from a validated external configuration,
and its resolved configuration can be reproduced from realization metadata.

### Phase 4 — Shared wireless mathematics

#### Implementation

1. Implement dB/amplitude/power conversions with explicit naming.
2. Implement degree/radian conversion and angle wrapping.
3. Implement spherical/cartesian direction conversion.
4. Implement complex-normal sampling with a documented variance convention.
5. Implement exponential and one-ring correlation matrices.
6. Implement positive-semidefinite matrix square roots with backend-specific
   stable algorithms.
7. Implement wavelength, Doppler, and geometry helpers.
8. Keep constants such as the speed of light in one shared physical-constants
   module.

#### Tests

Create separate files under `tests/unit/wireless/math/` for conversions,
angles, random distributions, correlation, and geometry. Include analytic
values, round trips, periodic boundaries, Hermitian/PSD properties, empirical
complex variance, and backend parity.

#### Acceptance gate

All later models can depend on these primitives without reimplementing unit
conversion, angle handling, random sampling, or correlation logic.

### Phase 5 — Noise and basic fading models

#### Implementation

1. Implement AWGN as an independent noise source.
2. Define and document noise variance per complex dimension.
3. Implement Rayleigh block fading as a CIR sampler.
4. Implement flat MIMO fading generation.
5. Implement Kronecker spatial correlation.
6. Implement per-column receive correlation.
7. Implement flat-channel signal application separately from generation.
8. Add external profiles for model sizes, precision, normalization, and
   correlation selection; do not embed experimental defaults in constructors.

#### Tests

Create:

- `test_awgn.py`;
- `test_rayleigh_block_fading.py`;
- `test_flat_fading.py`;
- `test_spatial_correlation.py`;
- `test_apply_flat_channel.py`.

Verify mean, variance, circular symmetry, block constancy, independence across
batch entries, covariance after correlation, identity-correlation behavior,
matrix multiplication against hand calculations, seeded replay, configuration
validation, and both backends.

#### Acceptance gate

An end-to-end flat MIMO simulation can sample a realization, apply it to a
signal, add controlled AWGN, and replay from the same seed.

### Phase 6 — Time and OFDM response processing

#### Implementation

1. Implement discrete time-lag bound calculation.
2. Implement subcarrier-frequency generation with documented DC and even/odd
   FFT conventions.
3. Implement CIR-to-time-response conversion with configurable interpolation
   or pulse-shaping kernel; start with the specified sinc formulation.
4. Implement CIR-to-OFDM-response conversion.
5. Implement time-domain signal application.
6. Implement OFDM signal application.
7. Implement optional response normalization as an explicit transform.
8. Keep AWGN outside the application operators; provide convenience composition
   only after the independent pieces are tested.
9. Preserve gradients through deterministic PyTorch transforms and operators.

#### Tests

Create:

- `test_time_lags.py`;
- `test_subcarrier_frequencies.py`;
- `test_cir_to_time.py`;
- `test_cir_to_ofdm.py`;
- `test_apply_time_channel.py`;
- `test_apply_ofdm_channel.py`;
- `test_response_normalization.py`.

Use analytic fixtures:

- a zero-delay unit path;
- a pure-delay path;
- two paths with known constructive/destructive frequency response;
- a constant flat MIMO path;
- a time-varying phase ramp with known Doppler.

Test shape, broadcasting, complex dtype, convolution boundaries, OFDM phase,
normalization energy, PyTorch gradients, and NumPy/PyTorch parity.

#### Acceptance gate

The same sampled CIR can be transformed and applied repeatedly in both time and
OFDM domains without resampling the channel.

### Phase 7 — TDL models

#### Implementation

1. Define `TDLConfig` and schema.
2. Transcribe scalable TDL profiles from the applicable standard into versioned
   resources with independent review.
3. Transcribe fixed-delay conformance profiles into their own specification
   directory.
4. Validate profile/frequency/bandwidth/delay-spread combinations.
5. Implement normalized tap powers and delays.
6. Implement NLoS tap generation.
7. Implement LoS components and K-factor handling.
8. Implement sum-of-sinusoids time variation or another explicitly documented
   standard-compatible fading process.
9. Implement speed-to-Doppler conversion.
10. Implement optional transmit, receive, and joint spatial correlation.
11. Register TDL model names and external profiles.

#### Tests

Create:

- `test_tdl_config.py`;
- `test_tdl_resources.py`;
- `test_tdl_delays_and_powers.py`;
- `test_tdl_sampling.py`;
- `test_tdl_doppler.py`;
- `test_tdl_spatial_correlation.py`;
- `test_tdl_reference_statistics.py`.

Test table row counts and checksums, total normalized power, RMS delay spread,
fixed versus scalable delays, LoS power split, zero-speed constancy, Doppler
spectrum bounds, antenna dimensions, seed replay, both backends, and optional
external statistical parity.

#### Acceptance gate

Every supported TDL profile produces a validated `CIRBatch`, passes analytic
profile checks, and meets documented statistical tolerances on PyTorch and
NumPy.

### Phase 8 — Antennas, coordinates, and polarization

#### Implementation

1. Define coordinate-frame and orientation conventions before writing formulas.
2. Implement spherical basis vectors and local/global coordinate transforms.
3. Implement an antenna-element interface returning polarized field patterns.
4. Implement isotropic and standard 3GPP antenna elements.
5. Implement panel and multi-panel array geometry.
6. Implement single and dual polarization and cross/slant arrangements.
7. Implement array-element positions in wavelength-normalized and metric form.
8. Keep plotting in a separate optional visualization module.
9. Store named antenna layouts in external profiles.

#### Tests

Create:

- `test_coordinate_transforms.py`;
- `test_antenna_elements.py`;
- `test_antenna_patterns.py`;
- `test_antenna_panels.py`;
- `test_antenna_arrays.py`;
- `test_polarization.py`.

Test orthonormal bases, forward/inverse rotations, boresight gain, attenuation
limits, symmetry, element spacing, panel indexing, polarization orthogonality,
orientation behavior, backend parity, and known table/figure sample points.

#### Acceptance gate

Antenna field responses and element locations are independently validated and
can be consumed without special cases by the CDL coefficient generator.

### Phase 9 — CDL models

#### Implementation

1. Define `CDLConfig` and schema.
2. Transcribe CDL-A through CDL-E tables from the standard with review metadata.
3. Implement cluster delays, powers, mean angles, spreads, XPR, and LoS K-factor
   loading.
4. Implement the standard ray offset table.
5. Implement random angle coupling and initial phases.
6. Implement deterministic topology creation for the single-link CDL case.
7. Implement antenna/polarization-aware coefficient generation.
8. Implement delay normalization, motion, and Doppler.
9. Return both `CIRBatch` and optional `RayBatch` diagnostics.

#### Tests

Create:

- `test_cdl_config.py`;
- `test_cdl_resources.py`;
- `test_cdl_rays.py`;
- `test_cdl_coefficients.py`;
- `test_cdl_sampling.py`;
- `test_cdl_doppler.py`;
- `test_cdl_reference_statistics.py`.

Test resource integrity, cluster/ray counts, power normalization, angular
offsets, random permutations, LoS/NLoS behavior, array response, velocity
phase evolution, delay normalization, polarization, backend parity, and
statistical reference behavior.

#### Acceptance gate

CDL-A through CDL-E generate physically consistent MIMO CIRs for uplink and
downlink antenna assignments on both backends.

### Phase 10 — Topology and scenario geometry

#### Implementation

1. Define immutable topology inputs: BS/UT positions, velocities,
   orientations, indoor state, site identity, and optional fixed LoS state.
2. Compute 2D/3D distances and LoS arrival/departure angles.
3. Implement uplink/downlink views without mutating a shared `RayBatch`.
4. Implement topology generators separately from channel models.
5. Define a `Scenario` interface for probability, applicability, parameter
   lookup, and path loss.
6. Implement RMa, UMa, UMi, InH, and InF geometry constraints one at a time.
7. Validate carrier frequency, heights, distances, room/factory parameters,
   and indoor/outdoor combinations.

#### Tests

Create `test_topology_geometry.py`, `test_topology_generation.py`, and one
scenario-configuration test file per scenario. Test hand-calculated geometry,
angle quadrants, collocated-point errors, uplink/downlink transposition,
topology reproducibility, scenario bounds, and both backends.

#### Acceptance gate

Every supported scenario accepts a validated topology and produces consistent
link geometry without sampling LSPs or rays yet.

### Phase 11 — Path loss and large-scale parameters

#### Implementation

For each scenario, in this order: UMi, UMa, RMa, InH, InF:

1. Transcribe versioned scenario parameter tables with review metadata.
2. Implement LoS probability.
3. Implement basic LoS and NLoS path loss.
4. Implement outdoor-to-indoor loss where applicable.
5. Implement shadow-fading parameters.
6. Implement means, standard deviations, and cross-correlation matrices for
   delay spread, angular spreads, shadow fading, K-factor, ZSA, and ZSD.
7. Implement spatial correlation over topology.
8. Sample and return a typed `LargeScaleParameters` object.
9. Make path loss and shadow fading independently enableable without changing
   the underlying random draws unexpectedly.

#### Tests

For each scenario create:

- `test_<scenario>_resources.py`;
- `test_<scenario>_los_probability.py`;
- `test_<scenario>_pathloss.py`;
- `test_<scenario>_large_scale.py`.

Also create shared tests for LSP correlation and spatial filtering. Verify
standard example points, breakpoint continuity, LoS/NLoS ordering where
specified, O2I components, empirical log-domain moments, cross-correlation,
distance correlation, masks, reproducibility, and backend parity.

#### Acceptance gate

All scenarios reproduce their tabulated formulas and expected LSP statistics
before any small-scale rays are generated.

### Phase 12 — Cluster and ray generation

#### Implementation

1. Implement cluster delay generation and sorting.
2. Implement cluster powers and per-cluster shadowing.
3. Implement azimuth and zenith cluster angles.
4. Implement per-ray angle offsets.
5. Implement XPR and initial phases.
6. Implement random ray coupling with explicit random streams.
7. Implement strongest-cluster identification and subclustering information.
8. Support differing valid cluster counts through masks rather than sentinel
   physics values.
9. Implement optional spatially correlated random fields as a separate layer.

#### Tests

Create focused files for delays, powers, angles, XPR/phases, coupling,
subclustering, masks, and spatial correlation. Verify sorted nonnegative
delays, normalized power, configured distributions, angular wrapping, coupling
permutations, strongest-cluster selection, invalid-link masking,
reproducibility, backend parity, and published statistical targets.

#### Acceptance gate

Every supported scenario can generate a complete, validated `RayBatch`
independent of antenna coefficient generation.

### Phase 13 — System-level coefficients and channel assembly

#### Implementation

1. Implement field-pattern evaluation at each ray angle.
2. Implement polarization coupling matrices.
3. Implement array phase offsets.
4. Implement Doppler phase evolution.
5. Combine ray contributions into clusters.
6. Implement specified subclustering for strongest clusters.
7. Implement deterministic LoS contribution and K-factor scaling.
8. Apply path loss and shadow fading as explicit final transforms.
9. Assemble `SystemLevelWirelessChannel` from scenario, LSP, ray, antenna,
   coefficient, and loss components.
10. Return optional intermediate diagnostics without changing random draws.

#### Tests

Create files for field response, polarization matrix, array phase, Doppler,
subcluster combination, LoS combination, loss application, and full system
assembly. Use reduced hand-computable fixtures before statistical tests. Verify
axis order, uplink/downlink reciprocity assumptions, masks, optional diagnostic
invariance, backend parity, and end-to-end reference statistics.

#### Acceptance gate

RMa, UMa, UMi, InH, and InF produce batched antenna-aware CIRs that can be
passed directly to the Phase 6 time and OFDM transforms.

### Phase 14 — Blockage and spatial consistency

#### Implementation

1. State the exact supported subset of the standard in capability metadata.
2. Implement self-blocking geometry.
3. Implement non-self-blocking Model A.
4. Implement screen-based Model B.
5. Apply blockage to ray amplitudes at the documented stage.
6. Implement snapshot spatial consistency for LoS state, LSPs, and small-scale
   variables.
7. Keep stateful mobility evolution and temporal blocker evolution disabled
   until they have their own state model and validation suite.
8. Ensure enabling diagnostics or unrelated features does not reorder random
   streams.

#### Tests

Create:

- `test_self_blocking.py`;
- `test_blockage_model_a.py`;
- `test_blockage_model_b.py`;
- `test_spatial_consistency_fields.py`;
- `test_spatial_consistency_metrics.py`;
- `test_feature_random_stream_isolation.py`.

Test boundary geometry, zero-loss outside blockers, attenuation inside
blockers, coherent power effects, correlation versus distance, repeatability,
feature isolation, both backends, and documented calibration curves.

#### Acceptance gate

Supported blockage and spatial-consistency capabilities have explicit tests and
explicitly documented omissions; no unsupported dynamic behavior is implied.

### Phase 15 — Dataset support, serialization, and metrics

#### Implementation

1. Define a versioned realization serialization schema.
2. Store arrays in a format suitable for large datasets while keeping metadata
   human-readable.
3. Include resolved configuration, seed/stream identity, backend, precision,
   model version, standard version, and software version.
4. Implement finite and repeatable streaming CIR datasets.
5. Implement deterministic shuffle buffers with injected RNG.
6. Implement replay as a channel sampler.
7. Implement RMS delay spread and circular angular spread.
8. Implement singular-value and frequency-selectivity metrics.
9. Provide scalar semantic path export for selected examples.

#### Tests

Create tests for schema validation, round-trip equality, large-array chunking,
stream exhaustion/repetition, deterministic shuffle, replay, corruption and
version errors, analytic metrics, masked paths, and both backends.

#### Acceptance gate

A generated realization can be stored, loaded on either backend, replayed, and
analyzed without changing its physical meaning or axis order.

### Phase 16 — Validation, optimization, and release readiness

#### Implementation

1. Run the complete fast suite on both backends.
2. Run slow statistical and reference suites with recorded environments.
3. Compare PyTorch and NumPy deterministic outputs at double precision.
4. Profile PyTorch first on representative CPU and CUDA workloads.
5. Optimize allocations, broadcasting, and vectorization without changing the
   public contract.
6. Add `torch.compile` compatibility only where benchmarks show benefit.
7. Profile and optimize NumPy separately; do not route it through PyTorch.
8. Produce capability documentation listing models, standard versions,
   validated ranges, optional features, and known limitations.
9. Add end-to-end examples using configuration files rather than constructor
   literals.
10. Freeze schema version 1 only after migration and replay tests pass.

#### Tests

Add end-to-end integration tests for:

- Rayleigh → time response → waveform;
- TDL → OFDM response → MIMO resource grid;
- CDL with polarized arrays;
- each system-level scenario;
- dataset generation → serialization → replay;
- PyTorch-produced dataset loaded into NumPy and the reverse.

Add non-blocking benchmark baselines for representative batch sizes and path
counts. Correctness tests remain mandatory; performance regressions are
reported separately until stable thresholds are established.

#### Acceptance gate

The documented supported feature matrix is fully exercised by repeatable test
commands, PyTorch remains the default and fastest-priority path, and NumPy is a
real independently executing backend.

## 9. Required test-file map

At minimum, the completed wireless implementation should have this systematic
test coverage:

| Area | Required permanent test files |
| --- | --- |
| Backends | registry, NumPy, PyTorch, random streams, parity |
| Data contracts | CIR, responses, topology, LSP, clusters, rays, scalar conversion |
| Configuration | envelope, profiles, hashing, resources, one config test per model |
| Math | conversions, angles, distributions, correlation, geometry |
| Basic models | AWGN, Rayleigh, flat fading, correlation, flat application |
| Responses | time lags, subcarriers, CIR-to-time, CIR-to-OFDM, application, normalization |
| TDL | resources, delays/powers, sampling, Doppler, correlation, statistics |
| Antennas | transforms, elements, patterns, panels, arrays, polarization |
| CDL | resources, rays, coefficients, sampling, Doppler, statistics |
| Scenarios | geometry plus resource, LoS, path-loss, and LSP tests per scenario |
| Rays | delays, powers, angles, XPR, phases, coupling, masks, subclusters |
| Coefficients | fields, polarization, array phase, Doppler, LoS, losses, assembly |
| Added features | blockage A/B, self-blocking, spatial consistency, RNG isolation |
| Dataset | schema, round trip, streaming, shuffle, replay, cross-backend loading |
| Metrics | delay spread, angular spread, singular values, masking |
| Integration | flat, TDL, CDL, each scenario, serialization/replay |

Do not place unrelated model tests into a single large test module. Tests
should mirror the source structure closely enough that a future contributor can
find a component's validation suite immediately.

## 10. Definition of done for an individual component

A component is complete only when all applicable items are satisfied:

- [ ] Public interface and units are documented.
- [ ] Typed configuration and external profile exist.
- [ ] Unknown and invalid configuration values are rejected.
- [ ] PyTorch implementation is complete.
- [ ] NumPy implementation or backend-neutral kernel is complete.
- [ ] No global random state is used.
- [ ] Shape, axis, dtype, precision, and device behavior are validated.
- [ ] Unit tests cover nominal, boundary, and error behavior.
- [ ] Stochastic behavior has reproducibility and statistical tests.
- [ ] Deterministic behavior has NumPy/PyTorch parity tests.
- [ ] At least one analytic or standards-based physical check passes.
- [ ] Integration with upstream and downstream containers is tested.
- [ ] Configuration and realization provenance are serializable.
- [ ] Applicability limits and unsupported features are explicit.
- [ ] Test runtime is classified as fast, statistical, slow, or reference.

## 11. Recommended first implementation increment

The first coding increment should stop after a complete vertical slice rather
than beginning 3GPP work immediately:

1. Phase 0 dependency and test setup.
2. Phase 1 backend and RNG protocols.
3. The `CIRBatch` portion of Phase 2.
4. The minimum configuration work from Phase 3.
5. Complex-normal sampling from Phase 4.
6. Rayleigh block fading from Phase 5.
7. CIR-to-OFDM conversion and OFDM application from Phase 6.
8. One end-to-end configured example and replayable seeded test.

This slice validates the backend abstraction, tensor contract, configuration
boundary, stochastic reproducibility, and generate/transform/apply separation
before the substantially larger TDL, CDL, and system-level work begins.
