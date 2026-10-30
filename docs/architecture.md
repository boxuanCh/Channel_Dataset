# Architecture

## 1. Goals and boundaries

The library represents and simulates physical-layer communication channels without forcing every medium into the same propagation equations. It standardizes the lifecycle and the data exchanged between layers, while allowing RF, acoustic, underwater, and optical implementations to express their own physics.

It is not initially responsible for transmitters, receivers, coding, networking, or link-level protocols. Those systems consume channel realizations and responses exposed by this package.

## 2. Layered design

```text
Configuration files (YAML/JSON)
            |
            v
  Loader + typed validation
            |
            v
Registry / typed model construction
            |
            v
 ChannelModel.realize(request)
            |
            v
 Fixed realization (for wireless, a CIRBatch)
            |
            +--> channel response / impulse response
            +--> apply to waveform
            +--> serialize dataset record
```

### Core domain

`core` contains concepts that retain the same meaning across media:

- `PropagationPath`: one resolved propagation contribution (often called a ray in geometric models).
- `PathCluster`: a group of paths sharing a physical scattering mechanism or statistical origin.
- `ExecutionContext`: numerical backend, precision, and device, kept separate
  from physical model settings and random seeds.
- `SimulationContext`: medium-neutral endpoint and environment inputs retained
  for models that need scalar geometry.
- `ChannelRealization`: a fixed, reusable sample from a channel model.
- `ChannelModel`: the interface implemented by all model families.

The word *path* is preferred in the shared API because not every channel is ray-based. A wireless or optical implementation may expose `Ray` as a family-specific specialization or alias when it carries extra geometry.

### Components

Reusable algorithms belong in `components`: geometry, path-loss laws, stochastic distributions, filtering, interpolation, Doppler processing, and noise sources. Components should have narrow interfaces and be composable by model implementations.

### Channel families

Each directory under `channels` contains a cohesive family:

- `wireless`: multipath RF, terrestrial/satellite links, MIMO, standardized tapped-delay models.
- `acoustic`: air acoustics, room impulse responses, outdoor sound propagation.
- `underwater`: sound-speed profiles, absorption, surface/bottom interactions, time variation.
- `fiber`: attenuation, dispersion, nonlinear effects, polarization-mode dispersion.
- `optical`: free-space optical and related atmospheric effects.

A family owns its typed configuration classes and physics. It depends on `core`, not on sibling families.

## 3. Principal abstractions

### Model versus realization

A `ChannelModel[RequestT, RealizationT]` is a configured probability law or
deterministic solver. Calling `realize(request)` produces a fixed realization.
Wireless models accept a `ChannelSamplingRequest`, which carries batch/time
dimensions, an `ExecutionContext`, and an injected `RandomStream`. Signal
application is a separate operator rather than a model method. Keeping these
separate supports:

- repeatable application to multiple signals;
- comparisons between receiver algorithms on the same channel;
- dataset serialization and replay;
- deterministic unit testing.

### Propagation path

The initial shared representation contains complex gain (including magnitude and phase), delay, Doppler shift, optional departure/arrival directions, and metadata. Medium-specific properties—polarization matrices, bounce geometry, acoustic boundary identifiers, fiber modes—should be represented in typed extensions before they become widely used. The metadata field is an escape hatch for forward compatibility, not the primary modeling API.

### Batched wireless contracts

Wireless hot paths use immutable Python containers around backend-native arrays.
`CIRBatch`, time/flat/OFDM responses, `TopologyBatch`,
`LargeScaleParameters`, `ClusterBatch`, and `RayBatch` validate canonical axes,
precision, backend, device, masks, and shared dimensions. Link selection keeps
singleton semantic axes and returns array views. Explicit `select_paths`
conversion creates scalar `PathCluster` and `PropagationPath` objects only for
diagnostics or serialization, never implicitly in simulation.

A sampled `CIRBatch` is reusable across representation domains. Deterministic
transforms produce either a sinc-interpolated `TimeChannelResponse` or an
`OFDMChannelResponse`; separate operators then apply those responses to a
waveform or resource grid. Time responses carry their integer lag origin so
convolution does not depend on out-of-band state. Optional power normalization
is also an explicit transform, and AWGN remains outside every application
operator.

Topology generation is similarly separate from channel sampling. A
`TopologyBatch` owns endpoint state, while `LinkGeometry` is a deterministic
pairwise derivation. Scenario objects validate whether that geometry is in
scope before later phases sample LoS state, path loss, large-scale parameters,
or rays. Small-scale initialization consumes topology plus typed large-scale
parameters and emits a complete antenna-independent `RayBatch`; coefficient
generation remains downstream. Reciprocal link and ray views return new
immutable containers and do not mutate shared diagnostics.

IEEE 802.11 models add `WLANChannelSamplingRequest` as composition rather than
changing the shared request. The TGn correlated-tap kernel owns A-F cluster
overlap, PAS covariance, Bell-spectrum evolution, and optional large-scale
components. TGac and TGah apply separate versioned bandwidth transformations.
TGax indoor scenarios compose that accepted kernel; TGax outdoor large-scale
resources remain separate from TR 38.901 even where equations look similar.

TR 38.811 models similarly compose `NTNChannelSamplingRequest` from the shared
sampling controls and an `NTNTopologyBatch`. Caller-supplied ECEF trajectories
remain authoritative. Deterministic geometry records absolute slant delay and
platform Doppler, while the nested `CIRBatch` carries synchronized excess
delays. Large-scale attenuation is retained as named components, and an
explicit NTN OFDM transform controls whether absolute propagation-delay phase
is included. This keeps ordinary CIR transforms reusable without silently
creating enormous time-domain lag windows.

TR 37.885 urban models compose `V2XSceneSamplingRequest` and
`V2XChannelSamplingRequest` around the same execution and waveform contracts.
The scene owns lane-bound vehicle trajectories on a slower mobility clock;
each channel request selects one snapshot for a fixed waveform window.
`V2XChannelState` carries condition episodes and correlated shadowing to the
next snapshot, while the returned `CIRBatch` remains compatible with every
ordinary time/OFDM transform and operator. Standard reference scenes and
controlled variations share types but are explicitly distinguished in
realization diagnostics.

### Cluster

A cluster provides identity and shared metadata for a collection of paths. It does not assume a particular clustering algorithm. Family models may add delay/angle spreads or physical interaction data in typed specializations.

### Units

The domain uses SI units internally. Ambiguous scalar names are avoided: `delay_s`, `frequency_hz`, `distance_m`, and similar names carry unit suffixes. A future quantities package may be added at API boundaries if unit conversion becomes a recurring source of errors.

## 4. Configuration design

Configuration documents use a common envelope:

```yaml
schema_version: 1
model:
  family: wireless
  name: example_tdl
  parameters:
    # model-owned fields
```

The loader performs safe parsing, strict envelope validation, and deterministic
profile resolution. The registry selects the implementation by `(family,
name)`, and that implementation validates the fully resolved `parameters` into
a typed, immutable configuration object.

Rules:

- No scientifically meaningful default should be hidden in implementation code.
- Defaults belong in a named configuration profile and are covered by validation tests.
- Values must carry units in field names or structured `{value, unit}` representations.
- Unknown fields should fail validation unless a schema explicitly allows extension fields.
- Schema revisions are integers. Breaking changes require a migration or a new supported version.
- Every produced dataset record includes the resolved configuration and provenance.

YAML is used for human-authored profiles; JSON remains supported for generated
configuration. Both map to the same in-memory envelope. Installed resources use
`package://` references rooted at `channel_dataset.resources`. External
profiles use `file:` references; relative paths are anchored to the containing
configuration file and never implicitly to the current working directory.

Profiles may extend one parent. Resolution is parent → child → envelope
override. Mapping values merge recursively; sequences and scalars replace the
inherited value. Cycles, model-identity mismatches, unknown fields, and
unsupported schema versions are errors.

An optional top-level `execution` section specifies backend, precision,
device, and seed. It is excluded from the physical model configuration digest,
so equivalent model settings retain the same identity across execution
backends.

The common envelope and profile format are described by
`configs/schema/model-config-v1.schema.json` and
`configs/schema/model-profile-v1.schema.json`. Model-specific parameter schemas
compose with the envelope. Copies are bundled with the installed package for
runtime tooling and are checked against the repository versions in tests.

## 5. Extension example

To add a model:

1. Create `channels/<family>/<model>.py`.
2. Define an immutable typed parameter object and its validation.
3. Implement `ChannelModel.realize` against a typed request and fixed
   realization type.
4. Register a builder for the `(family, name)` key.
5. Add versioned profiles under `src/channel_dataset/resources/profiles/<family>/`.
6. Add deterministic seeded tests and physical sanity checks.
7. Document the source of equations and validity range.

## 6. Planned increments

1. Complete configuration validation and model registry.
2. Implement shared response representations and waveform application.
3. Add a small reference wireless tapped-delay-line model end to end.
4. Add an acoustic impulse-response model to test whether abstractions are genuinely cross-medium.
5. Add dataset serialization, provenance, and batch simulation orchestration.
6. Expand to standardized and high-fidelity family-specific models with optional dependencies.

The first two concrete families should be deliberately different (for example RF and underwater acoustics); this exposes leaky abstractions earlier than adding several similar RF models.
