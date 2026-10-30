# Acoustic Channel Research and Implementation Plan

## 1. Scope and conclusion

The repository should keep its existing distinction between:

- `channels/acoustic`: airborne free-field, outdoor, and room acoustics; and
- `channels/underwater`: underwater acoustic propagation and communication.

There is no single acoustic equivalent of the 3GPP or IEEE WLAN channel-model
suites. The standards landscape is split:

- Airborne outdoor propagation has standardized attenuation calculations in
  ISO 9613-1:1993 and ISO 9613-2:2024.
- Room standards in ISO 3382 define measurement procedures and metrics, not a
  generative room-channel model. The image-source method is the common
  deterministic model for rectangular rooms, but it is not an ISO standard.
- Underwater communications has a standardized waveform/protocol in NATO
  STANAG 4748 (JANUS), but no widely accepted standardized channel models.
  BELLHOP is a de facto deterministic propagation tool, statistical models
  such as Qarabaqi--Stojanovic are research models, and replay libraries such
  as Watermark and the 2026 Underwater Acoustic Channel Library are emerging
  comparison benchmarks.

The implementation should therefore label models as `standard`,
`literature_model`, `de_facto_tool_adapter`, or `measurement_replay`. It must
not describe every common model as standardized.

## 2. Source inventory

### 2.1 Airborne and room acoustics

| Source | Status | What it supplies | Implementation consequence |
| --- | --- | --- | --- |
| [ISO 9613-1:1993](https://www.iso.org/standard/17426.html) | Current international standard, confirmed in 2026 | Frequency-, temperature-, humidity-, and pressure-dependent atmospheric absorption | Implement as a reusable deterministic component and a free-field channel model |
| [ISO 9613-2:2024](https://www.iso.org/standard/74047.html) | Current international standard | Octave-band outdoor level prediction including geometrical divergence, atmospheric absorption, ground effect, reflections, and screening | Treat as an engineering band-level realization, not a phase-resolved waveform channel |
| [ISO 3382-1:2009](https://www.iso.org/standard/40979.html) and [ISO 3382-2:2008](https://www.iso.org/standard/36201.html) | Current measurement standards | Impulse-response-derived room metrics and reverberation-time measurement | Use for validation/metrics; do not claim they define a room simulator |
| [Allen and Berkley, 1979](https://doi.org/10.1121/1.382599) | Canonical literature model | Image-source room impulse response for rectangular rooms | First room model |

The public ISO summaries are sufficient to establish scope but not to claim an
exact conforming implementation. Work on ISO 9613 equations and reference
cases is gated on access to the complete licensed editions. Copyrighted prose,
figures, and tables must not be copied into the repository.

### 2.2 Underwater acoustics

| Source | Status | What it supplies | Implementation consequence |
| --- | --- | --- | --- |
| [Stojanovic and Preisig, 2009](https://doi.org/10.1109/MCOM.2009.4752682) | Canonical survey | Frequency-dependent attenuation, time-varying multipath, wideband Doppler, colored noise, and absence of a standard fading model | Defines the minimum underwater feature set |
| [BELLHOP / Acoustics Toolbox manual](https://oalib-acoustics.org/website_resources/AcousticsToolbox/manual/node61.html) | De facto scientific tool, not a channel standard | Gaussian beam/ray tracing, eigenrays, arrivals, sound-speed profiles, and boundary interactions | Add an optional adapter rather than reimplementing a full ray solver first |
| [NPL seawater absorption guidance](https://resource.npl.co.uk/acoustics/techguides/seaabsorption/) | National metrology guidance summarizing primary formulas | Francois--Garrison and Ainslie--McColm absorption formulas and validity ranges | Implement formula components with explicit range validation |
| [Wenz, 1962](https://doi.org/10.1121/1.1909155) | Canonical measured-noise model | Frequency-dependent turbulence, shipping, wind/surface, and thermal noise regimes | First analytic colored-noise component |
| [Qarabaqi and Stojanovic, 2013](https://doi.org/10.1109/JOE.2013.2278787) | Widely used physics-based statistical model | Nominal multipath plus large- and small-scale displacement, fading, and Doppler | Implement after deterministic paths and wideband time variation exist |
| [Underwater Acoustic Channel Library, 2026](https://arxiv.org/abs/2609.03207) and [project site](https://uwa-channels.github.io/) | New open measurement-replay benchmark; explicitly not yet a standard | Band-limited measured time-varying impulse responses, arrays, colored Gaussian/impulsive noise, and replay | Support as an external checksummed dataset, not bundled package data |
| [JANUS / STANAG 4748 status](https://www.cmre.nato.int/ar25/maritime-unmanned-systems-enablers/) | Standardized digital underwater signaling | Waveform/interoperability context | Useful later for an end-to-end example, but not a channel model |

## 3. Relationship to wireless channel models

Acoustic and RF channels can both be expressed as linear time-varying systems,
and both use paths, delays, complex gains, arrays, frequency responses, and
Doppler. That shared mathematics is enough to reuse `ChannelModel`, execution
contexts, random streams, provenance, serialization patterns, and selected
signal operators. It is not enough to reuse every wireless data structure.

| Property | Terrestrial RF model tendency | Airborne acoustic tendency | Underwater acoustic tendency |
| --- | --- | --- | --- |
| Propagation speed | About 3e8 m/s | About 343 m/s near room conditions | About 1500 m/s, varying with temperature, salinity, and pressure/depth |
| Delay and latency | Nanoseconds per metre | Milliseconds per metre | Fractions of a millisecond per metre; operational links can have seconds of latency |
| Per-path response | Often approximated by one complex gain over a narrow band | Wall and air losses can be frequency dependent | Each path can be a strongly frequency-dependent filter |
| Doppler | Often modeled as a frequency shift over a narrow fractional bandwidth | Motion can be material for ultrasonic/audio links | Fractional Doppler and time dilation/compression are central; different paths may have different scales |
| Multipath | Statistical TDL/CDL models are common | Deterministic early reflections plus dense late reverberation | Surface/bottom/refraction paths with long, time-varying delay spreads; responses may be sparse or diffuse |
| Medium | Usually treated as locally homogeneous apart from scenario loss | Temperature, humidity, pressure, wind, and boundaries matter | Sound-speed profiles refract paths; bathymetry, surface state, and seabed geoacoustics matter |
| Noise | Thermal AWGN is a common first model | Environmental and device noise may be colored | Ambient noise is colored and can be directional, correlated, or impulsive |
| Field quantity | Vector electromagnetic field with polarization | Scalar acoustic pressure for the first implementation | Scalar pressure; no RF polarization, but projector/hydrophone directivity matters |
| Standardization | Mature 3GPP/IEEE/ITU model suites | Standard loss and measurement methods, but few full communication-channel models | No widely accepted channel-model suite as of 2026 |

For a fixed distance, sound is roughly five to six orders of magnitude slower
than radio. Consequently, absolute delay is a first-class channel output, not
normally a phase-only correction. Acoustic signals are also commonly wideband
in the fractional sense even when their absolute bandwidth is small, so a
narrowband Doppler-shift approximation cannot be the only supported operator.

## 4. Public contracts before model implementation

### 4.1 Requests and environment types

Add family-owned immutable types instead of unstructured dictionaries:

```text
channels/acoustic/
  requests.py
  types.py
  environments.py
channels/underwater/
  requests.py
  types.py
  environments.py
```

`AcousticSamplingRequest` and `UnderwaterSamplingRequest` should contain:

- execution context and injected random stream;
- batch/time dimensions and waveform sample rate;
- source and receiver positions/trajectories with explicit coordinate frame;
- source and receiver element geometry where arrays are requested;
- requested carrier/band or explicit frequency grid; and
- optional request identifier.

Air environments should type pressure, temperature, relative humidity, sound
speed or its calculation policy, room geometry, surfaces, and wind/ground
conditions. Underwater environments should type the sound-speed profile,
bathymetry, sea surface, seabed layers, source/receiver depth, and water
chemistry. A caller-supplied sound-speed profile is authoritative; empirical
sound-speed formulas are opt-in helpers with documented validity ranges.

### 4.2 Acoustic response types

Do not make `channels/acoustic` or `channels/underwater` import wireless
`CIRBatch`. Add medium-appropriate containers while retaining the same leading
link and array axes:

- `AcousticTransferFunctionBatch`: complex response on explicit time and
  frequency grids, axes
  `[batch, rx, rx_element, tx, tx_element, time, frequency]`;
- `AcousticTVIRBatch`: sampled time-varying impulse response, axes
  `[batch, rx, rx_element, tx, tx_element, output_time, lag]`;
- `AcousticPathBatch`: optional diagnostic path delays, directions, bounce
  identities, Doppler scale, and path mask; and
- `BandLevelRealization`: octave/third-octave attenuation or received levels
  for noncoherent engineering models such as ISO 9613-2.

The transfer response should state whether it is a normalized pressure ratio
or an absolute calibrated response. Never silently mix normalized channel
gain, sound pressure level, projector source level, and hydrophone sensitivity.
Absolute noise PSD should use `Pa^2/Hz`; metadata must record reference
pressures when decibel quantities are exposed.

### 4.3 Operators

Implement independently testable operators for:

1. LTI convolution with fractional-delay filtering;
2. LTV/TVIR application using block processing;
3. wideband path time scaling/resampling;
4. transfer-function sampling on arbitrary frequency grids; and
5. explicit addition of colored or impulsive noise.

Channel realization and noise generation remain separate, as in the wireless
family. Projector, microphone, and hydrophone responses should also be optional
composable components rather than hidden channel losses.

## 5. Model roadmap

### Phase A: foundations and free-field references

Implement the response types, geometry, transforms, application operators, and
serialization first. Add deterministic free-field models in both media to
prove units and normalization:

- `acoustic/free_field`: direct delay, spherical spreading, optional source
  and receiver directivity;
- `underwater/analytic_link`: direct delay, configurable spreading exponent,
  and frequency-dependent absorption.

Components in this phase:

- atmospheric sound speed and ISO 9613-1 absorption;
- underwater constant and tabulated sound-speed profiles;
- Mackenzie or UNESCO sound-speed helper, clearly labeled empirical;
- Ainslie--McColm first, then Francois--Garrison seawater absorption; and
- spherical/cylindrical/practical spreading with the convention recorded in
  configuration.

Acceptance criteria include analytic delay, inverse-distance pressure,
monotonic absorption versus range, reference-value comparisons, backend
parity, finite gradients where promised, and serialization round trips.

### Phase B: rectangular-room image-source model

Implement `acoustic/shoebox_image_source` from Allen--Berkley:

- direct sound and configurable reflection order/duration cutoff;
- six independently configured surfaces;
- frequency-dependent surface absorption/reflection;
- ISO 9613-1 air absorption on every image path;
- source/microphone directivity hooks;
- fractional-delay impulse construction; and
- optional deterministic late-tail approximation only as a separately named
  hybrid profile.

Do not put random late reverberation inside the deterministic image-source
model. Validation should cover image locations, reflection counts, direct-path
reduction to free field, reciprocity for reciprocal endpoints, path arrival
times, energy decay, and reasonable agreement between measured response
metrics and Sabine/Eyring estimates within documented applicability limits.
Expose ISO 3382-style metrics only after their definitions and test fixtures
have been independently reviewed; do not label simulator output as ISO
conformant measurements.

### Phase C: outdoor airborne engineering models

Implement two deliberately different products:

1. `acoustic/iso9613_1_free_field` produces a coherent frequency response for
   atmospheric absorption plus geometric delay/spreading.
2. `acoustic/iso9613_2_2024` produces octave-band engineering attenuation or
   sound-pressure levels under the standard's stated meteorological and source
   assumptions.

The second model must return `BandLevelRealization`; octave-band attenuation
must not be converted into a synthetic complex CIR without an explicitly
separate approximation. Exact work is gated on the full standard, controlled
equation transcription, independent review, and published/reference test
cases. Profiles must expose ground, barrier, reflection, meteorology, and
source-directivity inputs rather than hiding defaults.

### Phase D: shallow-water deterministic multipath

Implement `underwater/two_boundary_multipath` as a native, testable bridge
between the analytic link and BELLHOP:

- direct, surface-reflected, and bottom-reflected paths;
- configurable bounce order;
- complex, frequency-dependent surface/bottom reflection coefficients;
- per-path spreading and absorption;
- path angles and bounce metadata; and
- optional endpoint motion with wideband Doppler scaling.

This is a geometry model, not a substitute for refracting ray tracing. Its
profiles should say `constant_sound_speed` or `straight_ray` explicitly.
Tests should check image geometry, phase reversal at a pressure-release
surface, path ordering, absorption per path length, and convergence as the
bounce cutoff grows.

### Phase E: optional BELLHOP arrival adapter

Implement `underwater/bellhop_arrivals` behind an optional extra and an
external executable boundary:

- typed environment-to-input rendering;
- safe subprocess execution in a per-run temporary directory;
- parsing of arrivals/eigenrays into `AcousticPathBatch` and transfer/TVIR
  responses;
- support for tabulated range/depth sound-speed profiles and typed boundaries;
- executable version, input digest, run options, and convergence settings in
  provenance; and
- actionable errors when BELLHOP is absent.

Do not vendor or translate the solver without a separate license review.
Parser tests should use small checked-in text fixtures. Mark solver comparisons
as optional integration tests and verify convergence by increasing beam count
or reducing step size, as recommended by the BELLHOP manual.

### Phase F: measurement replay and noise

Add `underwater/tvir_replay` with a generic manifest before adding a named data
source. The manifest should carry URI, content checksum, license, DOI/source,
band limits, sample rates, array geometry, calibration/normalization,
environment, and extraction method.

Then add a loader/profile layer for the Underwater Acoustic Channel Library:

- external opt-in download/cache; never silently download during import;
- preserve the published in-band validity restriction;
- reconstruct TVIR, phase, and delay drift using the documented representation;
- preserve multihydrophone spatial correlation; and
- keep the accompanying Gaussian or alpha-stable colored noise generator
  separate from channel replay.

Also implement a generic Wenz-style colored ambient-noise PSD component with
shipping and wind/sea-state inputs. Treat Wenz curves as broad empirical
baselines, not a site-specific guarantee. Validate PSD slopes/levels against
published fixtures, covariance positive semidefiniteness, seeded replay, and
impulsive distribution quantiles.

### Phase G: physics-based stochastic underwater model

Implement a versioned `underwater/qarabaqi_stojanovic_2013` model only after
the wideband operators and deterministic nominal paths are stable:

- nominal path geometry from the native shallow-water model or imported
  BELLHOP arrivals;
- large-scale displacement and path-gain variation;
- small-scale scattering, path-specific Doppler, and gain variation;
- surface and bottom reflection effects;
- configuration profiles matching published experiments; and
- explicit state continuation for chunked time generation.

Distribution, coherence, Doppler-spread, and stationarity-region tests must use
published parameter sets or author-provided reference outputs. A seeded test
alone is not physical validation.

## 6. Configuration, registry, and package layout

Proposed layout:

```text
configs/
  acoustic/
    free_field/
    iso9613/
    rooms/
  underwater/
    analytic/
    shallow_water/
    bellhop/
    replay/
src/channel_dataset/
  channels/acoustic/
    config.py
    environments.py
    requests.py
    types.py
    propagation.py
    rooms/image_source.py
    outdoor/iso9613.py
    transforms/
    operators/
    metrics.py
  channels/underwater/
    config.py
    environments.py
    requests.py
    types.py
    absorption.py
    sound_speed.py
    boundaries.py
    analytic.py
    shallow_water.py
    bellhop/
    replay/
    statistical/
    noise/
    transforms/
    operators/
tests/
  unit/acoustic/
  unit/underwater/
  integration/acoustic/
  integration/underwater/
  reference/acoustic/
  reference/underwater/
  statistical/underwater/
```

Stable registry keys should be descriptive and source-specific, for example:

```text
(acoustic, free_field)
(acoustic, shoebox_image_source)
(acoustic, iso9613_1_free_field)
(acoustic, iso9613_2_2024)
(underwater, analytic_link)
(underwater, two_boundary_multipath)
(underwater, bellhop_arrivals)
(underwater, tvir_replay)
(underwater, qarabaqi_stojanovic_2013)
```

Every profile and realization must record schema version, model version,
standard or paper version, configuration digest, units, random stream, backend,
software version, and any external solver/dataset identity.

## 7. Recommended delivery order

The minimum useful vertical slices are:

1. Shared acoustic response contracts and an airborne free-field model.
2. Shoebox image-source room channel with time-domain application.
3. Underwater analytic link plus seawater absorption and colored noise.
4. Straight-ray shallow-water multipath with wideband Doppler.
5. BELLHOP arrivals adapter.
6. Underwater Acoustic Channel Library replay.
7. ISO 9613-2 engineering outdoor prediction after standards access/review.
8. Qarabaqi--Stojanovic statistical time variation.

This order exercises the repository with genuinely different media early,
delivers common room and underwater baselines quickly, and postpones the two
highest-risk tasks: exact standards conformance and calibrated stochastic
time-variation.

## 8. Definition of done for each model

A model is complete only when it has:

- strict typed configuration and JSON schema with no hidden scientific
  defaults;
- at least one named external profile and documented validity range;
- `ChannelModel.realize` returning a reusable fixed realization;
- separate waveform application and noise;
- deterministic seed/stream tests where stochastic;
- analytic physical-sanity tests and independent reference comparisons;
- NumPy and PyTorch CPU parity for native numerical kernels;
- serialization/replay with complete provenance;
- public API docstrings and one executable example; and
- an honest status label that distinguishes standards conformance, literature
  reproduction, tool interoperability, and measurement replay.
