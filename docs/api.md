# Python API reference

The public API uses NumPy-style terminology and explicit tensor-axis contracts.
All numerical functions preserve the selected backend: NumPy calls return
`numpy.ndarray` values and PyTorch calls return `torch.Tensor` values.

## Backends and execution

### `ExecutionContext`

```python
from channel_dataset.core import ExecutionContext

execution = ExecutionContext(
    backend="torch",       # "torch" or "numpy"
    precision="single",   # "single" or "double"
    device="cpu",
)
backend = execution.array_backend
stream = execution.random_stream(seed=20260923)
```

PyTorch is the default backend. NumPy supports CPU execution only. Precision
maps to the following dtype pairs:

| Precision | Real | Complex |
| --- | --- | --- |
| `single` | `float32` | `complex64` |
| `double` | `float64` | `complex128` |

### `RandomStream`

Streams provide `uniform`, `normal`, `complex_normal`, `integers`,
`bernoulli`, `permutation`, and `fork`. They never modify global random state.
A fork is derived from the root seed and full label lineage:

```python
coefficients = stream.fork("coefficients")
first = coefficients.complex_normal((32,), variance=1.0, dtype="complex64")

# Re-deriving the same label replays the same component stream.
replayed = stream.fork("coefficients")
second = replayed.complex_normal((32,), variance=1.0, dtype="complex64")
```

For complex-normal samples, `variance` means `E[abs(z)**2]`; the real and
imaginary components each have variance `variance / 2`.

## Configuration and model construction

`load_config(path)` accepts YAML and JSON. `parse_config_document(mapping)`
validates an already-decoded document. Unknown fields are errors. Model
parameters may be written directly or inherited from a packaged/file profile:

```yaml
schema_version: 1
model:
  family: wireless
  name: rayleigh_block_fading
  profile: package://profiles/wireless/rayleigh_2x2.yaml
  parameters:
    average_power_linear: 0.5  # highest-precedence override
```

Profile parents resolve first, then child mappings, then envelope overrides.
Nested mappings merge recursively; scalar and sequence values replace their
inherited values. Packaged references are independent of the working directory,
while relative `file:` references resolve against the containing configuration
file.

```python
from channel_dataset.channels.wireless import register_wireless_models
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry

config = load_config("configs/wireless/rayleigh_block_fading_profile.yaml")
registry = ModelRegistry()
register_wireless_models(registry)
model = registry.create(config)
```

`register_wireless_models` is the convenient default for applications that
load arbitrary checked-in wireless configurations. Family-specific functions
such as `register_rayleigh_model` and `register_tdl_model` remain available
when an application intentionally exposes a smaller model set.

`ConfigEnvelope.model_digest` is a canonical SHA-256 digest of the physical
model identity and parameters. Backend, device, precision, seed, and experiment
metadata do not alter it.

`config.to_document()` returns a JSON-serializable envelope containing the
fully resolved parameters rather than a profile reference. A realization saves
the same resolved parameters in its metadata; use
`ConfigEnvelope.from_realization_metadata(metadata)` to reconstruct and verify
the physical configuration. `load_resource_document(...)` and
`resource_checksum(...)` provide direct access to packaged profiles and schemas.

## Shared wireless mathematics

`channel_dataset.channels.wireless.math` contains backend-neutral primitives
used by later fading and scenario models:

- explicitly named power/amplitude dB conversions;
- degree/radian conversion and half-open angle wrapping;
- azimuth/elevation ↔ Cartesian direction conversion;
- exponential and uniformly distributed one-ring correlation matrices;
- native NumPy/PyTorch principal PSD matrix square roots;
- wavelength, Doppler, distance, line-of-sight, and closing-speed helpers;
- shared SI constants, including the exact speed of light.

Functions receiving an `ExecutionContext` create native arrays with that
context's precision and device. Functions receiving an existing native array
preserve its backend. Positive closing speed means two endpoints are
approaching, so it produces a positive Doppler shift.

```python
from channel_dataset.channels.wireless.math import (
    exponential_correlation,
    matrix_sqrt_psd,
    wavelength_m,
)

correlation = exponential_correlation(4, 0.7, execution)
correlation_root = matrix_sqrt_psd(correlation)
wavelength = wavelength_m(3.5e9)
```

## Sampling a channel

```python
from channel_dataset.channels.wireless import ChannelSamplingRequest

execution = config.execution.context()
request = ChannelSamplingRequest(
    batch_size=8,
    num_time_samples=14,
    sampling_frequency_hz=15_360_000.0,
    execution=execution,
    random_stream=execution.random_stream(config.execution.seed),
    request_id="example",
)
cir = model.realize(request)
```

IEEE 802.11 models compose this request with optional topology rather than
changing its semantics. See {doc}`ieee80211_models` for TGn, TGac, TGah, and
the supported TGax subset, including exact revisions and review status.

`RayleighBlockFading.realize` returns one zero-delay path whose coefficient is
constant over the requested time samples. Recreating the request stream from
the same seed replays the realization.

## Flat MIMO fading and AWGN

`FlatMIMOFading` samples responses with axes
`[batch, rx, rx_antenna, tx, tx_antenna]`. Its profiles select per-coefficient
or per-link power normalization and one of three correlation modes:

- `none` for independent entries;
- `kronecker` for separable receive/transmit correlation;
- `per_column_receive` for a distinct receive covariance per transmit antenna.

`apply_flat(response, signal)` accepts leading signal axes
`[batch, tx, tx_antenna]`, sums the transmit dimensions, and preserves any
trailing sample dimensions. AWGN remains an independent component:

```python
from channel_dataset.channels.wireless.noise import (
    AdditiveWhiteGaussianNoise,
    AWGNConfig,
)
from channel_dataset.channels.wireless.operators import apply_flat

received = apply_flat(flat_response, transmitted)
noise = AdditiveWhiteGaussianNoise(AWGNConfig(noise_variance_linear=0.1))
noisy_received = noise.apply(received, request.random_stream)
```

`noise_variance_linear` means `E[abs(n)**2]` per complex sample. Independent
real and imaginary components each have half that variance. Noise and fading
use stable child-stream labels, so enabling noise cannot reorder fading draws.

## Tensor contracts

Public containers are frozen dataclasses. Their array fields retain native
backend behavior and mutability.

| Container or field | Canonical axes |
| --- | --- |
| `CIRBatch.coefficients` | `[batch, rx, rx_ant, tx, tx_ant, path, time]` |
| `CIRBatch.delays_s`, `path_mask` | `[batch, rx, tx, path]` |
| `TimeChannelResponse.coefficients` | `[batch, rx, rx_ant, tx, tx_ant, output_time, lag]` |
| `OFDMChannelResponse.coefficients` | `[batch, rx, rx_ant, tx, tx_ant, symbol, subcarrier]` |
| `FlatChannelResponse.coefficients` | `[batch, rx, rx_ant, tx, tx_ant]` |
| Topology endpoint vectors | `[batch, endpoint, xyz-or-orientation]` |
| Topology link masks | `[batch, rx, tx]` |
| `LargeScaleParameters` fields | `[batch, rx, tx]` |
| `ClusterBatch` fields | `[batch, rx, tx, cluster]` |
| `RayBatch` ray fields | `[batch, rx, tx, cluster, ray]` |

Constructors validate ranks, shared dimensions, dtype/precision, backend,
device, masks, finiteness, and nonnegative quantities where required.

### Link selection

Every batched container provides `select_link(batch_index, rx, tx)`. Selection
preserves singleton semantic axes and uses basic NumPy/PyTorch slicing, so the
result shares storage with the source arrays.

```python
one_link = cir.select_link(batch_index=0, rx=0, tx=0)
assert one_link.shape[:2] == (1, 1)
```

`select_paths(ray_batch, batch_index, rx, tx)` is an explicit diagnostic
adapter. It converts only the chosen link into scalar `PathCluster` and
`PropagationPath` objects; batched simulation never creates one Python object
per ray.

## OFDM processing

```python
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.transforms import (
    cir_to_ofdm,
    subcarrier_frequencies,
)

frequencies = subcarrier_frequencies(64, 15_000.0, execution)
response = cir_to_ofdm(cir, frequencies)

signal = backend.ones(
    (8, 1, 2, 14, 64),
    dtype=execution.complex_dtype,
    device=execution.device,
)
received = apply_ofdm(response, signal)
```

The transmit grid axes are `[batch, tx, tx_ant, symbol, subcarrier]`. The
received grid axes are `[batch, rx, rx_ant, symbol, subcarrier]`.
`cir_to_ofdm` evaluates

```text
H(f, t) = sum_p a_p(t) exp(-j 2 pi f tau_p)
```

and ignores masked paths. `apply_ofdm` sums over transmitter and
transmit-antenna axes. Neither operation adds noise, and both preserve PyTorch
gradients.

## Time-domain processing and response normalization

Time-domain conversion uses normalized sinc interpolation to map physical path
delays onto integer sample lags. Automatic bounds include a configurable
truncation margin; callers may instead provide exact inclusive bounds.

```python
from channel_dataset.channels.wireless.operators import apply_time
from channel_dataset.channels.wireless.transforms import (
    cir_to_time,
    normalize_response,
    time_lag_bounds,
)

lag_min, lag_max = time_lag_bounds(
    max_delay_s=3e-6,
    sampling_frequency_hz=request.sampling_frequency_hz,
)
time_response = cir_to_time(
    cir,
    request.sampling_frequency_hz,
    lag_min=lag_min,
    lag_max=lag_max,
)
time_response = normalize_response(time_response)
received_waveform = apply_time(time_response, transmitted_waveform)
```

`apply_time` accepts transmit axes `[batch, tx, tx_ant, input_time]` and returns
`[batch, rx, rx_ant, output_time]`. It evaluates
`y[n] = sum_l h[n, l] x[n-l]`, treating samples beyond the supplied waveform as
zero. `TimeChannelResponse.lag_min` records the physical lag represented by the
first element of the final axis, and `lag_max` is derived from its length.

`normalize_response` accepts flat, time, and OFDM response containers. It
scales every `[batch, rx, tx]` link independently to the requested mean squared
coefficient magnitude. Normalization is never implicit in conversion or signal
application. AWGN remains a separate operation.

(api-3gpp-tdl)=
## 3GPP tapped-delay-line models

`TappedDelayLine` supports the scalable TDL-A through TDL-E profiles from
3GPP TR 38.901 V16.1.0 and the fixed TDLA30, TDLB100, TDLC300, and TDLC60
conformance profiles from 3GPP TS 38.101-4 V16.1.0. Standard tables are packaged
as versioned resources, separate from execution and experiment configuration.

```python
from channel_dataset.channels.wireless.tr38901 import register_tdl_model
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry

envelope = load_config("configs/wireless/example_tdl.yaml")
registry = ModelRegistry()
register_tdl_model(registry)
tdl = registry.create(envelope)
cir = tdl.realize(request)
```

Scalable profiles multiply normalized delays by `delay_spread_s`. Fixed
profiles retain the conformance table's physical delays. All profile powers are
normalized to unit total power. TDL-D and TDL-E keep their specular and diffuse
zero-delay components separate and optionally accept an aggregate Ricean
`k_factor_db` adjustment.

Diffuse taps use a seeded finite sum-of-sinusoids approximation of the classical
Jakes spectrum, bounded by the speed-derived maximum Doppler frequency. The LoS
component uses the standard `0.7 * maximum_doppler_hz` spectral line. Spatial
correlation can be disabled, constructed as receive/transmit Kronecker factors,
or provided as one validated joint antenna covariance matrix.

The resource metadata records its specification, version, table, applicability
range, source URL, transcription date, and review status. The current tables
are marked `pending_independent_review`; checksums prevent silent edits while
that external review remains outstanding.

## Antenna coordinates, polarization, and arrays

The antenna API follows the 3GPP TR 38.901 coordinate convention. Zenith is
measured from positive z, azimuth is measured from positive x toward positive
y, and an `Orientation` applies intrinsic bearing, downtilt, and mechanical
slant rotations in that order. All public angles are radians.

```python
from math import pi

from channel_dataset.channels.wireless.antennas import (
    AntennaPanel,
    MultiPanelArray,
    Orientation,
    ThreeGPPAntennaElement,
)

panel = AntennaPanel(
    rows=8,
    columns=8,
    polarization_slant_angles_rad=(-pi / 4, pi / 4),
    element=ThreeGPPAntennaElement(),
)
array = MultiPanelArray(panel)
positions = array.positions_m(3.5e9, execution)
field = array.field_pattern(
    0,
    zenith_rad=pi / 2,
    azimuth_rad=0,
    execution=execution,
    orientation=Orientation(bearing_rad=0),
)
```

Panels lie in the local yz-plane and point along positive x. Element indexing
is row, column, polarization with polarization fastest; multi-panel indexing
adds panel row and panel column as the outer dimensions. Positions are
available in wavelengths or metres. `PolarizedFieldPattern.theta` and `.phi`
are field amplitudes in the requested global spherical basis, and their
squared magnitudes sum to linear power gain.

`ThreeGPPAntennaElement` implements the Table 7.3-1 vertical and horizontal
cuts, attenuation caps, 8 dBi boresight gain, and the clause 7.3.2
angle-independent linear-polarization model. Packaged named layouts can be
loaded with `load_antenna_layout`; layout YAML is strict and versioned against
`antenna-layout-v1.schema.json`. Plotting lives separately in the optional
`antennas.visualization` module and imports Matplotlib only when called.

## 3GPP clustered-delay-line models

`ClusteredDelayLine` implements CDL-A through CDL-E from 3GPP TR 38.901
V16.1.0. Each diffuse cluster expands to the 20 Table 7.5-3 rays, with seeded
angle coupling, four polarization phases, XPR, antenna element responses,
array-location phases, and motion-induced Doppler. CDL-D and CDL-E additionally
include their tabulated specular component; `k_factor_db` can replace the
aggregate table K-factor while retaining normalized total power.

```python
from channel_dataset.channels.wireless.tr38901 import register_cdl_model

envelope = load_config("configs/wireless/example_cdl.yaml")
registry = ModelRegistry()
register_cdl_model(registry)
cdl = registry.create(envelope)
cir, rays = cdl.realize_with_rays(request)
topology = cdl.topology(request)
```

The first implementation is deliberately single-link: every batch entry has
one transmitter and one receiver. It supports either uplink or downlink by
assigning the configured base-station and terminal arrays to the corresponding
roles without mutating diagnostics. The returned `CIRBatch` aggregates the 20
rays into one path per standard cluster, while `RayBatch` retains the sampled
angles, XPR, and initial phases for inspection.

## Topology and scenario geometry

`RectangularTopologyGenerator` samples endpoint positions, receiver headings,
indoor state, and optional fixed LoS state from independent labeled streams.
It does not sample channel parameters. `compute_link_geometry` derives 2D/3D
distance plus global departure and arrival angles for every batch/receiver/
transmitter link and rejects collocated endpoints.

```python
from channel_dataset.channels.wireless import (
    RectangularTopologyConfig,
    RectangularTopologyGenerator,
    ScenarioConfig,
    compute_link_geometry,
    create_scenario,
)

generator = RectangularTopologyGenerator(
    RectangularTopologyConfig(1, 8, 200, 200, 10, 1.5, rx_speed_mps=3)
)
topology = generator.generate(request)
geometry = compute_link_geometry(topology)
scenario = create_scenario(ScenarioConfig("UMi", carrier_frequency_hz=3.5e9))
scenario.validate_topology(topology)
```

The scenario layer validates versioned RMa, UMa, UMi, InH, and InF frequency,
distance, height, room, factory, and indoor/outdoor constraints. Phase 11 adds
LoS probability and LoS/NLoS path-loss equations, explicit O2I loss,
cross-correlated log-domain large-scale parameters, and exponential spatial
filtering. The current indoor subsets are InH mixed office and InF sparse
clutter/low base station; capability metadata and resources state that scope.
`LinkGeometry.reverse()` and `reverse_ray_batch()` create reciprocal views
without mutating shared input objects.

```python
scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
lsp = scenario.sample_large_scale(
    topology,
    execution.random_stream(2026),
    path_loss_enabled=True,
    shadow_fading_enabled=True,
)
```

Path loss and shadow fading can be disabled independently for controlled
experiments. The sampler still draws the same LoS and seven Gaussian LSP
variables, so toggling either output does not perturb delay/angle spreads or
K-factor. Invalid links are retained through masks and have zeroed physics.

## Scenario cluster and ray generation

`ScenarioRayGenerator` converts topology and large-scale state into an
antenna-independent `RayBatch`. It generates sorted nonnegative cluster
delays, normalized powers, cluster and coupled ray angles, lognormal XPR, and
initial phases. RMa, UMa, UMi, InH, and InF select their LoS/NLoS cluster
counts from a versioned resource. Links with fewer clusters use Boolean masks;
invalid entries never contain sentinel physical values.

```python
from channel_dataset.channels.wireless import ScenarioRayGenerator

ray_generator = ScenarioRayGenerator("UMi")
rays = ray_generator.generate(
    topology,
    lsp,
    request.random_stream.fork("small_scale"),
)
subclusters = ray_generator.subcluster_plan(rays)
```

Component streams are derived by label, so diagnostics and unrelated parent
draws cannot reorder the result. `ExponentialSpatialField` is a separate
opt-in layer for spatially correlated Gaussian variables; ray generation
itself remains a snapshot operation.

## System-level channel assembly

`SystemLevelWirelessChannel` joins the scenario, LSP, ray, antenna, and
coefficient layers for a fixed topology. The coefficient engine evaluates
polarized element fields, XPR coupling, array phase, and Doppler; it then
combines rays, splits the two strongest clusters into three subclusters, adds
the deterministic LoS contribution, and applies path loss plus shadowing as a
final voltage transform.

```python
from channel_dataset.channels.wireless import SystemLevelWirelessChannel
from channel_dataset.channels.wireless.antennas import load_antenna_layout

layout = load_antenna_layout(
    "profiles/wireless/antenna_isotropic_siso.yaml"
)
channel = SystemLevelWirelessChannel(
    ScenarioConfig("UMi", 3.5e9), topology, layout, layout
)
cir = channel.realize(request)
```

`realize_with_diagnostics` additionally returns the LSP, `RayBatch`, and
optional coefficient terms. Enabling those diagnostics does not alter random
draws. The resulting canonical `CIRBatch` can be passed directly to
`cir_to_time` or `cir_to_ofdm`.

## Blockage and snapshot spatial consistency

Phase 14 supports the TR 38.901 V16.1.0 self-blocking portrait and landscape
angular regions, snapshot non-self-blocking Model A, and stationary-screen
Model B. Blockage is carried as per-ray decibel loss and applied to voltage
coefficients before ray combination. Multiple blockage models can be supplied
to `SystemLevelWirelessChannel` and their decibel losses accumulate.

`SnapshotSpatialConsistency` supplies repeatable exponentially correlated LoS,
LSP, and small-scale Gaussian fields. It is deliberately stateless: stateful
mobility evolution, temporal blocker evolution, and dynamic blocker
birth/death are not supported. The machine-readable `CAPABILITIES` object
lists this exact boundary, so callers do not infer dynamic behavior from the
snapshot APIs.

## Dataset persistence, replay, and metrics

`save_cir` writes a schema-version-1 realization directory with readable JSON
metadata and checksummed, non-pickled NPY array chunks. `load_cir` reconstructs
the same canonical axes on either NumPy or PyTorch, making stored realizations
backend-portable. Existing nonempty destinations are never overwritten.

```python
from channel_dataset.channels.wireless import load_cir, save_cir

save_cir("dataset/example-0001", cir, chunk_size=64)
replayed = load_cir("dataset/example-0001", request.execution)
```

`StreamingCIRDataset` provides finite or explicitly repeating indexed streams;
`shuffle_stream` adds a deterministic bounded shuffle using an injected random
stream, and `ReplayChannel` exposes compatible stored CIRs through the common
model interface. Analysis helpers compute RMS delay spread, circular angular
spread, per-tone MIMO singular values, and normalized frequency selectivity
without switching numerical backends.

## Error behavior

- `TypeError` indicates an incompatible Python type or array dtype.
- `ValueError` indicates invalid values, shapes, ranks, backend/device mixing,
  non-finite values, or configuration fields.
- `IndexError` indicates an out-of-range semantic-axis selection.
- `ImportError` explains how to install PyTorch when the default backend is
  unavailable.

See the individual Python docstrings with `help(symbol)` for complete parameter,
return, exception, and notes sections.
