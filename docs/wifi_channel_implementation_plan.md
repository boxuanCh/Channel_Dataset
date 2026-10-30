# IEEE 802.11 WLAN Channel Model Implementation Plan

## 1. Purpose

This document defines the implementation roadmap for standardized IEEE 802.11
wireless-LAN channel models in Channel Dataset. It covers the TGn channel
models, the TGac addendum, TGah sub-1-GHz models, and the TGax indoor/outdoor
extensions. It also identifies which existing infrastructure is reused, which
physics must be implemented independently, and the validation required before
a model is reported as supported.

TGn, TGac, and TGax are IEEE 802.11 task-group models, not 3GPP models. The
TGax outdoor work draws on UMi/UMa models with cellular origins, but its source
of truth is the IEEE evaluation document and the versions it references. The
existing TR 38.901 implementation must not be relabeled or silently substituted
for an IEEE model.

The implementation is standards-first and independent:

- Standards documents, approved IEEE contributions, and their supporting
  material define physical behavior.
- Numeric tables and formulas are transcribed into versioned data resources
  with source, clause, units, review state, and checksum metadata.
- Vendor implementations may be used as optional behavioral oracles, but
  their code, tests, table layout, comments, or internal design must not be
  copied.
- Every stochastic component uses an injected `RandomStream`; production code
  never uses process-global random state.
- A sampled channel remains separate from waveform application and AWGN.

## 2. Normative and informative sources

The first implementation baseline is:

1. [IEEE 802.11-03/940r4, TGn Channel Models](https://mentor.ieee.org/802.11/dcn/03/11-03-0940-04-000n-tgn-channel-models.doc).
2. [IEEE 802.11-09/0308r12, TGac Channel Model Addendum](https://mentor.ieee.org/802.11/dcn/09/11-09-0308-12-00ac-tgac-channel-model-addendum-document.doc).
3. [IEEE 802.11-09/0569r0, TGac Channel Model Supporting Material](https://mentor.ieee.org/802.11/dcn/09/11-09-0569-00-00ac-tgac-channel-model-addendum-supporting-material.doc).
4. [IEEE 802.11-11/0968r4, TGah Channel Model Text](https://mentor.ieee.org/802.11/dcn/11/11-11-0968-04-00ah-channel-model-text.docx).
5. [IEEE 802.11-14/0882r4, TGax Channel Model Document](https://mentor.ieee.org/802.11/dcn/14/11-14-0882-04-00ax-tgax-channel-model-document.docx).

The exact revision is part of every profile identity. A later revision is added
as a new resource namespace and must not change an existing profile in place.
Before transcribing tables, confirm that each document is the final task-group
revision intended by the project and preserve a local provenance record without
redistributing a source document unless its terms permit that redistribution.

## 3. Scope

### 3.1 Included in the first program

The program includes:

1. TGn delay profiles A through F.
2. SISO and MIMO channel realizations.
3. Per-tap clustered power, delay, AoA, AoD, and angular-spread parameters.
4. Tap-specific transmit and receive spatial correlation.
5. NLoS Rayleigh and LoS Rician behavior.
6. Standards-defined time evolution and environmental Doppler.
7. TGn breakpoint path loss and lognormal shadow fading as optional,
   independently testable large-scale components.
8. Standards-defined fluorescent-light and moving-object Doppler options where
   applicable.
9. TGac bandwidth extension, up to 8-by-8 MIMO, and MU-MIMO behavior.
10. TGah sub-1-GHz operation, bandwidth-dependent delay-profile behavior, and
    TGah link/deployment path-loss scenarios.
11. TGax indoor reuse of TGn/TGac and TGax-specific outdoor/system-level
    propagation.
12. NumPy and PyTorch CPU support, with CUDA-compatible PyTorch array paths.
13. CIR, time-domain, and OFDM output through existing public contracts.
14. Serialization, replay, metrics, examples, and Sphinx documentation.

### 3.2 Deferred or separate programs

The following are not part of this plan's acceptance boundary:

- TGad/TGay 60-GHz directional and quasi-deterministic models;
- TGbf sensing channels and target/scene evolution;
- TGbb optical wireless channels;
- protocol, MAC, scheduling, rate control, packet error, or traffic models;
- transmitter and receiver RF impairments unless a channel-model document
  explicitly makes one part of the propagation evaluation model;
- full ray tracing or scene-engine integration;
- bit-identical stochastic samples across NumPy and PyTorch;
- packet-to-packet stateful filtering in the initial TGn release.

TGad/TGay should become a separate `ieee80211_mmwave` plan. Reusing the common
backend and response contracts is appropriate, but forcing those models through
the TGn correlated-tap engine is not. Neither model is an acceptance criterion
for this plan.

## 4. Reuse boundary

### 4.1 Fully reusable infrastructure

The following existing capabilities are reused without changing their public
semantics:

- `ArrayBackend`, NumPy/PyTorch implementations, and backend registry;
- `RandomStream` and SHA-256-derived child streams;
- `ExecutionContext` and realization provenance;
- configuration envelopes, profile inheritance, resource loading, and model
  registry;
- `ChannelModel[RequestT, RealizationT]`;
- `CIRBatch` with axes
  `[batch, rx, rx_antenna, tx, tx_antenna, path, time]`;
- `TimeChannelResponse`, `OFDMChannelResponse`, and flat responses;
- `cir_to_time`, `cir_to_ofdm`, response normalization, and subcarrier helpers;
- time-domain, flat, and OFDM signal application;
- AWGN as a separate noise source;
- CIR persistence, streaming, replay, and corruption detection;
- RMS delay spread, frequency selectivity, and MIMO metrics;
- Sphinx API generation, pytest organization, markers, and CI structure.

In a WLAN `CIRBatch`, the `path` axis represents resolvable profile taps. Tap
delays are shared by antenna pairs, which matches the existing delay shape
`[batch, rx, tx, path]`. A `path_mask` supports mixed profiles or padded
multi-link batches.

### 4.2 Partially reusable infrastructure

The following components are useful but need IEEE-specific extensions:

- `TopologyBatch` can represent APs and stations, but multiuser WLAN generation
  needs a request that explicitly associates topology with channel sampling.
- Existing antenna panels cover common ULA and rectangular-array cases. An
  arbitrary-coordinate array type is required only if a normative evaluation
  case cannot be represented by the existing panel geometry.
- Matrix square roots, correlation utilities, and complex-normal sampling are
  reusable. TGn tap correlation matrices must be calculated from the IEEE
  angular model rather than from the current exponential-correlation shortcut.
- The 3GPP TDL implementation provides a lifecycle and testing pattern. Its
  standard tables, fading equations, correlation modes, and Doppler behavior
  are not reused as TGn physics.
- The current 3GPP scenario machinery may provide geometry helpers for TGax.
  IEEE-referenced UMi/UMa versions and calibration rules remain separate.

### 4.3 New physics

The following must be implemented specifically for IEEE 802.11:

- A-F profile resources and strict typed loaders;
- clustered tap construction and cluster-overlap accounting;
- truncated power-angular-spectrum integration and tap-specific spatial
  covariance;
- standards-defined temporal fading spectrum and filter/state generation;
- TGn LoS fixed component and K-factor handling;
- fluorescent and moving-object Doppler components;
- TGn breakpoint path loss and shadow fading;
- TGac delay/bandwidth transformation and multiuser rules;
- TGax-specific indoor penetration and outdoor scenario behavior.

The expected reuse is approximately 75-85 percent of framework infrastructure,
but only 40-55 percent of total delivery effort. Standards transcription,
physics, validation, and review remain new work. After TGn is complete, TGac is
expected to reuse 70-85 percent of the TGn engine. TGah is expected to reuse a
similar proportion of the accepted TGn/TGac small-scale implementation while
adding separate sub-1-GHz bandwidth and large-scale resources.

## 5. Target package and resource structure

```text
src/channel_dataset/channels/wireless/ieee80211/
  __init__.py
  requests.py                 WLAN-specific request composition
  profiles.py                 Typed profile resources and validation
  config.py                   TGn/TGac/TGah/TGax configuration objects
  spatial.py                  PAS integration and spatial covariance
  temporal.py                 Doppler spectra and fading processes
  large_scale.py              WLAN path loss and shadowing
  tgn.py                      TGn model assembly
  tgac.py                     TGac extensions and MU-MIMO assembly
  tgah.py                     TGah sub-1-GHz model assembly
  tgax.py                     TGax indoor/outdoor model selection
  diagnostics.py              Optional typed diagnostic outputs

src/channel_dataset/resources/
  schema/
    ieee80211-tgn-v1.schema.json
    ieee80211-tgac-v1.schema.json
    ieee80211-tgah-v1.schema.json
    ieee80211-tgax-v1.schema.json
    ieee80211-profile-v1.schema.json
  profiles/wireless/ieee80211/
    03_940r4/
      model_a.yaml
      model_b.yaml
      model_c.yaml
      model_d.yaml
      model_e.yaml
      model_f.yaml
    09_0308r12/
      tgac_base.yaml
      bandwidth_rules.yaml
      multiuser_rules.yaml
    11_0968r4/
      tgah_base.yaml
      bandwidth_rules.yaml
      pathloss_scenarios.yaml
    14_0882r4/
      indoor.yaml
      outdoor_umi.yaml
      outdoor_uma.yaml
      penetration.yaml

configs/schema/               Repository copies of public schemas
configs/wireless/ieee80211/   Runnable strict examples

tests/unit/wireless/ieee80211/
tests/integration/wireless/ieee80211/
tests/statistical/wireless/ieee80211/
tests/reference/wireless/ieee80211/
```

Do not create one large `wifi.py`. Profile parsing, spatial mathematics,
temporal generation, large-scale loss, and model assembly have different
validation responsibilities and should remain independently testable.

## 6. Public interfaces and data contracts

### 6.1 Model identities

Register stable keys:

```text
wireless / ieee80211_tgn
wireless / ieee80211_tgac
wireless / ieee80211_tgah
wireless / ieee80211_tgax
```

Model identity includes the IEEE document revision and selected profile in the
resolved configuration, but backend, device, batch size, and seed remain
excluded from the physical configuration digest.

### 6.2 Sampling request

Add an immutable composition type rather than changing the existing sampling
request:

```python
@dataclass(frozen=True, slots=True)
class WLANChannelSamplingRequest:
    sampling: ChannelSamplingRequest
    topology: TopologyBatch | None = None
```

Rules:

- Link-level TGn/TGac/TGah generation may omit topology when link distance and
  endpoint counts are fully specified by configuration.
- Path loss, shadowing, geometry-derived LoS state, TGac MU-MIMO, TGah
  deployment scenarios, and TGax system scenarios require topology.
- Topology batch size, backend, device, and precision must match `sampling`.
- A request does not silently resample the time axis.
- The model returns `CIRBatch`; transforms and application remain separate.

If testing shows that static link distance in configuration and topology create
ambiguous duplicate sources, remove static distance and require a minimal
topology for every large-scale calculation. There must be one authoritative
distance per link.

### 6.3 Profile types

Create immutable profile types:

```python
IEEE80211Profile
IEEE80211TapProfile
IEEE80211ClusterProfile
TGacBandwidthRule
TGacMultiUserRule
TGahBandwidthRule
TGahPathLossProfile
```

At minimum, tap/cluster profiles carry:

- tap index and delay in seconds;
- cluster identity;
- power in dB and resolved linear power;
- mean AoA/AoD in radians;
- receive/transmit angular spreads in radians;
- optional LoS/fixed-component classification;
- provenance reference to document, revision, table, and row.

Profile loading rejects unknown fields, duplicate tap indices, inconsistent
cluster membership, nonfinite values, negative delays, invalid angles, and
power tables that cannot be normalized within a documented tolerance.

### 6.4 Configurations

Create frozen `TGnConfig`, `TGacConfig`, `TGahConfig`, and `TGaxConfig`
objects. Shared fields include:

- exact standards revision;
- delay profile A-F;
- carrier frequency in hertz;
- transmitter and receiver antenna-layout references;
- link direction;
- environmental speed in metres per second;
- LoS selection policy;
- large-scale fading policy;
- optional environmental Doppler effects;
- profile/resource references and explicit overrides.

TGac adds:

- channel bandwidth in hertz;
- bandwidth-rule revision;
- number of users and selected user index where a per-user model is requested;
- MU-MIMO generation policy;
- transmission direction;
- applicable antenna-count limits.

TGah adds:

- sub-1-GHz carrier-frequency applicability;
- supported channel bandwidth in hertz;
- the document-defined delay-profile bandwidth rule;
- AP-to-station or station-to-station link type;
- indoor/outdoor and deployment-scenario identity;
- scenario-specific distance, height, and environmental parameters required by
  11/0968r4;
- scenario-specific path-loss and shadow-fading selection.

TGax adds:

- indoor or outdoor scenario identity;
- deployment/scenario parameters required by 14/0882r4;
- wall and floor counts or typed penetration geometry;
- outdoor UMi/UMa applicability fields;
- fast-moving-object Doppler switches where defined.

Unknown fields and unsupported combinations are errors. Scientifically
meaningful defaults live in named profiles. Constructors do not guess a model,
bandwidth, distance, LoS state, or normalization convention.

### 6.5 Diagnostics

The production result remains a `CIRBatch`. An optional diagnostic method may
return a frozen `WLANRealizationDiagnostics` containing:

- resolved tap and cluster power;
- transmit and receive covariance matrices per tap;
- selected LoS/K-factor state;
- large-scale loss components;
- temporal process parameters;
- stream lineage labels.

Diagnostics must not change random draws. Requesting diagnostics after a seeded
realization must reproduce the same CIR as generation without diagnostics.

### 6.6 Temporal continuity boundary

The first release generates a temporally consistent fixed window of
`num_time_samples`. It does not promise continuity across separate `realize`
calls, even when consecutive requests use adjacent times.

A later `WLANFadingProcess` may expose serializable filter state and an
`advance(num_samples)` operation. That API is accepted only after fixed-window
statistics and replay are complete. It must not introduce hidden mutable state
into `ChannelModel.realize`.

## 7. Algorithms

### 7.1 Profile resolution

1. Load a versioned A-F profile.
2. Validate delays, powers, clusters, angles, and spreads.
3. Convert units exactly once at the configuration boundary.
4. Resolve cluster overlap into per-tap cluster contributions.
5. Normalize only where required by the normative definition.
6. Preserve both source dB values and resolved linear powers for diagnostics.

Profile selection is deterministic and consumes no random numbers.

### 7.2 Spatial covariance

For each tap and each contributing cluster:

1. Construct the documented transmit and receive power-angular spectra.
2. Apply the documented truncation and normalization.
3. Integrate the phase difference over angle for every antenna pair using
   element positions in wavelengths.
4. Combine overlapping-cluster contributions using their tap powers.
5. Form Hermitian transmit and receive covariance matrices.
6. Project only small numerical negative eigenvalues caused by roundoff; reject
   materially non-PSD resources or equations.
7. Compute stable matrix square roots using the backend-neutral PSD-root helper.
8. Color IID complex fading independently per tap and link.

Quadrature order and tolerance are numerical execution controls, not physical
profile parameters. NumPy and PyTorch deterministic covariance calculations
must agree within precision-appropriate tolerance.

### 7.3 Small-scale fading

NLoS components use circular complex Gaussian processes with the documented
tap powers. LoS profiles add the documented fixed component and scale diffuse
and fixed components so the resulting Rician K-factor is correct without
changing total tap power.

Use stable child streams such as:

```text
wireless.ieee80211.<standard>.link.<batch>.<rx>.<tx>.tap.<tap>.diffuse
wireless.ieee80211.<standard>.link.<batch>.<rx>.<tx>.tap.<tap>.los_phase
wireless.ieee80211.<standard>.link.<batch>.<rx>.<tx>.shadow_fading
wireless.ieee80211.<standard>.link.<batch>.<rx>.<tx>.environmental_doppler
```

Adding diagnostics, disabling path loss, or requesting another independent
feature must not perturb unrelated draws.

### 7.4 Temporal fading

Implement the Doppler spectrum prescribed by the selected IEEE model rather
than reusing the 3GPP classical spectrum by name. The implementation may use a
filtered-Gaussian or independently validated spectral-synthesis method if it
reproduces the required autocorrelation and power spectral density.

Requirements:

- zero speed produces time-constant diffuse coefficients unless another
  enabled environmental effect introduces variation;
- generated processes have unit average power before tap scaling;
- sampling-frequency and aliasing constraints are validated;
- independent taps and clusters use independent child streams;
- LoS phase evolution is deterministic for fixed geometry and velocity;
- fluorescent-light and moving-object components are isolated optional
  components and have their own tests.

If the backend protocol lacks an operation needed by the selected algorithm,
add the narrow operation to both backends with protocol-conformance tests. Do
not route NumPy execution through PyTorch.

### 7.5 Large-scale propagation

Implement TGn path loss as a separate component:

1. Free-space behavior up to the profile breakpoint.
2. The documented post-breakpoint slope.
3. Continuous value at the breakpoint.
4. LoS/NLoS shadow-fading standard deviation from the selected profile.
5. Carrier-frequency dependence exactly as specified.

Return typed path-loss and shadowing diagnostics. Small-scale profile power is
generated independently and large-scale amplitude is applied once. AWGN and
receiver noise figures remain outside the channel model.

### 7.6 TGac bandwidth extension

Implement the TGac addendum as a transformation of a resolved TGn profile plus
TGac-specific rules:

- supported bandwidths and their exact delay-spacing transformation;
- sample-rate and resolvability validation;
- profile power preservation;
- antenna-count extension through 8-by-8 MIMO;
- MU-MIMO user and direction rules;
- any addendum changes to spatial or temporal behavior.

The bandwidth transformation must be a named, independently tested function.
Do not approximate it by changing only the OFDM subcarrier frequencies. The
physical delay profile and the evaluation waveform grid are separate concerns.

### 7.7 TGac MU-MIMO

Multiuser generation consumes a topology containing one or more APs and
stations. Implement exactly which components are shared or independent across
users according to 09/0308r12. In particular, tests must distinguish:

- AP-side spatial statistics;
- station-side spatial statistics;
- per-user fading streams;
- uplink/downlink reciprocity or direction-dependent construction;
- user indexing that does not change another user's random realization.

Do not implement MU-MIMO as a simple stack of independently generated SU-MIMO
channels unless the standard explicitly makes them independent.

### 7.8 TGah

TGah reuses the accepted TGn/TGac correlated-tap engine where 11/0968r4
retains the A-F profile behavior. It adds an IEEE-revision-specific layer for:

- sub-1-GHz carrier frequencies and corresponding wavelengths;
- supported narrow channel bandwidths;
- the document-defined bandwidth-dependent delay-profile transformation;
- TGah deployment and link types;
- AP-to-station and station-to-station path-loss behavior;
- scenario-specific shadowing and applicability ranges.

The TGah bandwidth transformation must be independent from the TGac transform,
even where both resolve to the same numeric scaling for a particular bandwidth.
Shared code is allowed only after tests demonstrate that the equations and
boundary conditions are identical.

Frequency, bandwidth, deployment scenario, endpoint heights, and link type are
validated before random sampling. Path loss and shadowing remain separable from
small-scale fading and are applied exactly once.

### 7.9 TGax

TGax work begins only after TGac acceptance.

Indoor TGax scenarios reuse the accepted TGn/TGac kernel and add the path-loss,
wall, floor, deployment, and fast-object behavior selected by 14/0882r4.

Outdoor TGax implements the exact UMi/UMa family and revision referenced by the
IEEE document. Existing TR 38.901 geometry and array helpers may be reused, but
parameter tables, LoS probability, path loss, LSP correlation, cluster counts,
and calibration assumptions require an explicit comparison report before any
existing implementation is shared.

The comparison report classifies every field as:

- identical and reusable;
- mathematically identical after unit conversion;
- similar but requiring an IEEE-specific resource;
- incompatible and requiring a separate algorithm.

## 8. Configuration and resource governance

Every standards resource includes metadata similar to:

```yaml
metadata:
  standards_body: IEEE
  document: 802.11-03/940
  revision: r4
  clause_or_table: Appendix C, Model B
  source_url: https://mentor.ieee.org/...
  units_reviewed: true
  transcription_review: pending_independent_review
  checksum: <sha256 of canonical physical table>
```

Requirements:

- Repository and packaged copies must match byte-for-byte or by canonical
  document digest.
- A table change requires a new checksum and review note.
- An independent reviewer checks every transcribed numeric row against the
  source before the profile loses `pending_independent_review` status.
- Source provenance and formula citations appear in docstrings and user docs.
- Apache-2.0 project licensing does not imply that an IEEE document may be
  redistributed; only project-authored configuration resources are packaged.

## 9. Testing strategy

### 9.1 Resource and configuration tests

Test:

- schema validity and strict unknown-field rejection;
- all A-F resources load from source and installed-package locations;
- TGah bandwidth and path-loss resources load from source and installed-package
  locations;
- exact delays, powers, cluster counts, K-factors, and breakpoint values;
- source metadata and stable checksums;
- configuration hashes are independent of backend, device, and seed;
- invalid bandwidth/profile/antenna combinations fail at the boundary;
- profile inheritance produces the same resolved parameters as an explicit
  configuration.

### 9.2 Data-contract tests

For every backend and precision, verify:

- canonical CIR axes and expected shape;
- coefficient, delay, and mask dtypes;
- backend and device consistency;
- finite coefficients and nonnegative delays;
- correct tap masks in mixed-profile batches;
- immutable metadata and storage-sharing link views;
- actionable failures for rank, shape, dtype, device, and topology mismatch.

### 9.3 Deterministic analytic tests

Use small hand-computable cases for:

- one-tap SISO generation;
- identity spatial covariance;
- two-element covariance at simple angles and zero angular spread;
- covariance Hermitian symmetry and positive semidefiniteness;
- Rician diffuse/fixed scaling;
- path-loss breakpoint continuity;
- bandwidth delay transformation;
- TGah narrowband delay transformation and path-loss reference points;
- zero-delay and pure-delay OFDM responses;
- MIMO application against explicit matrix multiplication.

### 9.4 Statistical tests

Statistical acceptance includes:

- diffuse complex mean near zero;
- per-tap empirical powers matching profile powers;
- Rayleigh amplitude quantiles or goodness-of-fit;
- Rician K-factor and amplitude-distribution checks;
- empirical transmit and receive covariance per representative taps;
- low cross-correlation between independent taps, clusters, links, and users;
- temporal autocorrelation at selected lags;
- Doppler PSD shape, bandwidth, symmetry, and integrated power;
- shadow-fading mean and standard deviation;
- fluorescent/moving-object spectral components;
- TGac MU-MIMO shared and independent correlation behavior.
- TGah path-loss, shadowing, and transformed-delay-profile statistics.

Use deterministic seeds, documented sample counts, and tolerances chosen from
sampling uncertainty. A failing statistical test must report the measured
statistic and expected interval. Avoid thresholds so tight that nominal Monte
Carlo variation makes CI flaky.

### 9.5 Cross-backend and gradient tests

- Deterministic profile, covariance, path-loss, and transform calculations must
  agree between NumPy and PyTorch.
- Each backend must replay its own stochastic realization exactly for the same
  seed and configuration.
- NumPy and PyTorch random samples are not required to be identical.
- PyTorch gradients must pass through deterministic CIR-to-time, CIR-to-OFDM,
  normalization, and application operations.
- Stochastic sampling and discrete profile selection are not promised to be
  differentiable.

### 9.6 Integration and replay tests

Add end-to-end cases for:

1. TGn profile YAML -> model -> CIR -> OFDM response -> resource grid.
2. TGn MIMO CIR -> time response -> waveform application.
3. LoS and NLoS seeded replay.
4. TGac 80/160-MHz profile generation.
5. TGac downlink MU-MIMO with at least two users.
6. TGah representative indoor, outdoor, and station-to-station links.
7. Save -> load -> cross-backend deterministic response comparison.
8. TGax indoor and outdoor representative scenarios.

Every integration result includes model revision, resolved configuration,
stream lineage, software version, backend, precision, and device.

### 9.7 Reference tests

Create project-owned reference vectors from hand calculations and independently
reviewed scripts. Optional comparison against a licensed third-party tool may
be documented, but CI must not require proprietary software.

Reference-vector metadata records:

- standards document and revision;
- configuration and seed;
- generation script version;
- expected quantities and tolerances;
- reviewer and review date.

### 9.8 Performance tests

Benchmark representative SISO, 4-by-4, and 8-by-8 configurations over realistic
time and tap counts. Track:

- realization time;
- peak memory;
- covariance-construction time;
- CIR-to-OFDM time;
- scaling with batch, antennas, taps, users, and time samples.

Initial performance tests are informational smoke tests. Regression thresholds
are introduced only after stable CI runners establish a baseline.

## 10. Implementation phases and gates

Each phase starts only after the previous phase's complete test selection passes
on NumPy and PyTorch CPU. Ruff and the full pre-existing suite are required at
every gate.

### Phase 0 - standards inventory and design freeze

Deliverables:

- Archive source identifiers, revision history, and clause/table inventory.
- Produce a parameter-by-parameter TGn/TGac/TGah/TGax extraction worksheet.
- Record ambiguities and obtain explicit resolution before implementation.
- Confirm redistribution policy for project-authored transcribed tables.
- Freeze naming, units, model keys, and scope.

Gate:

- Every planned parameter has a normative source or is marked derived with a
  documented equation.
- No unresolved ambiguity affects Model A/B implementation.

### Phase 1 - package, schemas, and typed profiles

Deliverables:

- Create the `ieee80211` package.
- Add strict profile/config schemas and typed parsers.
- Transcribe TGn A-F resources with provenance and checksums.
- Register model keys with constructors initially guarded as incomplete.
- Add example TGn configuration.

Gate:

- All resource/configuration tests pass.
- Independent review is complete for Models A and B; C-F may temporarily retain
  `pending_independent_review` while implementation continues.

### Phase 2 - TGn static SISO vertical slice

Deliverables:

- Implement A and B static SISO diffuse fading.
- Return validated `CIRBatch` objects.
- Reuse existing time/OFDM transforms and application.
- Add a seeded executable example and serialization replay.

Gate:

- Exact delays, powers, RMS delay spread, empirical tap powers, replay, OFDM,
  and cross-backend deterministic tests pass.
- No existing test regresses.

### Phase 3 - complete TGn profiles and spatial MIMO

Deliverables:

- Enable C-F profiles.
- Implement cluster overlap and PAS integration.
- Implement per-tap transmit/receive covariance and MIMO coloring.
- Support common ULA/rectangular spacing cases through antenna profiles.
- Add spatial diagnostics.

Gate:

- Resource review for A-F is complete.
- Analytic covariance, PSD, empirical covariance, antenna-count, and MIMO
  application tests pass on both backends.

### Phase 4 - TGn temporal, LoS, and large-scale behavior

Deliverables:

- Implement temporal fading spectrum.
- Implement Rician/fixed components.
- Implement breakpoint path loss and shadowing.
- Implement supported fluorescent-light and moving-object effects.
- Add applicability and aliasing validation.

Gate:

- Temporal ACF/PSD, K-factor, path-loss continuity, shadowing, spectral-effect,
  replay, and stream-isolation tests pass.
- TGn is documented as supported only after this gate.

### Phase 5 - TGac bandwidth and 8-by-8 MIMO

Deliverables:

- Add TGac configuration and versioned addendum resources.
- Implement the bandwidth delay transformation.
- Support 20/40/80/160-MHz cases defined by the source revision.
- Validate 8-by-8 generation and memory scaling.

Gate:

- Bandwidth reference vectors, delay/power preservation, 8-by-8 shapes,
  statistical checks, and end-to-end OFDM tests pass.

### Phase 6 - TGac MU-MIMO

Deliverables:

- Require and validate multiuser topology.
- Implement user/direction rules and shared AP-side statistics.
- Add order-independent per-user random streams.
- Add multiuser diagnostics and examples.

Gate:

- User-order invariance, shared/independent covariance, uplink/downlink,
  replay, masking, and multiuser OFDM tests pass.
- TGac is documented as supported only after this gate.

### Phase 7 - TGah sub-1-GHz models

Deliverables:

- Add `TGahConfig`, schema, model registration, and 11/0968r4 resources.
- Implement and validate the TGah bandwidth-dependent delay-profile rules.
- Add TGah link/deployment path loss and shadow fading.
- Reuse the accepted TGn/TGac small-scale engine only where the source
  comparison establishes equivalence.
- Add runnable indoor, outdoor, and station-to-station examples.

Gate:

- Carrier-frequency, bandwidth, transformed-delay, path-loss, shadowing,
  topology, replay, statistical, and end-to-end OFDM tests pass on NumPy and
  PyTorch.
- TGah is documented as supported only after all claimed link/deployment
  scenarios have independently reviewed resources.

### Phase 8 - optional stateful packet continuity

Deliverables:

- Decide from use cases whether fixed-window realization is insufficient.
- If required, design serializable `WLANFadingState` and
  `WLANFadingProcess` APIs.
- Preserve reset, seed, chunk-boundary, and batch semantics.

Gate:

- One long generation and multiple continued chunks agree within numerical
  tolerance.
- State serialization/replay and feature-stream isolation pass.

This phase may be skipped without weakening fixed-window TGn/TGac/TGah
compliance, provided the limitation remains explicit.

### Phase 9 - TGax indoor scenarios

Deliverables:

- Add TGax configuration and indoor resources.
- Reuse the accepted TGn/TGac small-scale kernel.
- Add walls, floors, penetration, deployment, and fast-object options specified
  by 14/0882r4.

Gate:

- Scenario-specific path loss, penetration, topology, Doppler, and end-to-end
  tests pass.

### Phase 10 - TGax outdoor scenarios

Deliverables:

- Complete the IEEE-versus-existing-3GPP comparison report.
- Implement or adapt the referenced outdoor UMi/UMa models.
- Add LoS, path loss, shadowing, spatial consistency, and calibration cases.

Gate:

- Reference formulas, statistical parameters, topology applicability,
  multi-link correlation, and calibration tests pass.
- TGax is documented as supported only after indoor and outdoor gates pass, or
  documentation explicitly names a narrower supported subset.

### Phase 11 - release hardening

Deliverables:

- Add public API docs, model-selection guide, examples, and capability matrix.
- Add migration and compatibility notes.
- Run NumPy-only, PyTorch CPU, combined, documentation, and optional CUDA CI.
- Record independent table/equation review status.
- Publish benchmark results and known limitations.

Gate:

- Ruff, full pytest, strict Sphinx, package build, installed-wheel import, and
  all supported CI matrices pass.
- No profile with `pending_independent_review` is described as independently
  verified.

## 11. CI matrix

At minimum, add or extend jobs for:

```text
NumPy-only:
  configuration, resource, unit, statistical, integration

PyTorch CPU:
  configuration, resource, unit, statistical, integration, gradients

Combined:
  backend parity, serialization cross-load, reference vectors

Documentation:
  sphinx -W --keep-going

Optional CUDA:
  marked smoke, shape/device, OFDM/application gradients
```

Statistical jobs may be separated for runtime, but a release gate runs them.
Marker registration should include `ieee80211`, `tgn`, `tgac`, `tgah`, and
`tgax` in addition to the existing backend/statistical/reference markers.

## 12. Documentation requirements

Documentation must include:

- a terminology note distinguishing IEEE 802.11 and 3GPP models;
- selection guidance for A-F profiles and supported applicability ranges;
- exact standards revision and review status;
- antenna, distance, speed, bandwidth, LoS, and normalization conventions;
- CIR axis semantics and how taps map to the path axis;
- fixed-window temporal-continuity limits;
- NumPy/PyTorch replay and parity boundaries;
- TGac multiuser topology examples;
- TGah bandwidth, link-type, and deployment-scenario examples;
- TGax supported scenario subset;
- a complete CIR-to-OFDM resource-grid example;
- known differences from popular third-party tools where their APIs add
  non-normative behavior.

Public functions and types require NumPy-style docstrings with shapes, units,
raises clauses, notes, and standards references.

## 13. Risks and mitigations

### Standards ambiguity

Mitigation: maintain an extraction worksheet and decision log; never resolve a
physical ambiguity through an undocumented implementation default.

### Table transcription error

Mitigation: canonical checksums, row-level tests, and independent review.

### Statistical-test flakiness

Mitigation: derive tolerances from sample uncertainty, use fixed seeds, report
measured values, and separate fast moments from slower distribution tests.

### Excessive memory for 8-by-8 multiuser batches

Mitigation: avoid expanding covariance matrices over time, cache immutable
profile covariance by execution context and array geometry, and benchmark
before adding compiled paths.

### Accidental 3GPP/IEEE parameter mixing

Mitigation: separate resource namespaces, model keys, typed configurations,
and provenance. Cross-family conversion must be explicit and tested.

### Proprietary implementation dependence

Mitigation: CI uses project-owned reference vectors and hand calculations;
third-party tools remain optional comparison oracles only.

### Stateful filtering conflicts with immutable realization lifecycle

Mitigation: complete fixed-window models first and, if needed, introduce a
separate serializable process abstraction rather than hidden mutable model
state.

## 14. Completion criteria

The program is complete when:

1. TGn A-F, TGac, TGah, and the explicitly claimed TGax subset are
   constructible through strict versioned configuration.
2. Every model returns a validated, reusable `CIRBatch` on NumPy and PyTorch.
3. Exact tables, analytical formulas, physical invariants, statistical
   distributions, spatial covariance, temporal spectra, and replay are covered
   by permanent tests.
4. Existing time/OFDM transforms, signal operators, persistence, and metrics
   work without WLAN-specific forks.
5. NumPy execution does not import PyTorch and no stochastic production path
   uses global RNG state.
6. The complete pre-existing and IEEE 802.11 test suites pass before each phase
   advances.
7. Strict Sphinx documentation and packaged resources build from an installed
   wheel.
8. Capability documentation states exact standards revisions, supported
   subsets, review status, and known limitations without overstating
   conformance.
