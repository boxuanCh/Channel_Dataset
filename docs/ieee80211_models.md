# IEEE 802.11 WLAN channel models

The `channel_dataset.channels.wireless.ieee80211` package implements the
fixed-window WLAN model lifecycle for these source revisions:

| Model key | Source revision | Supported subset |
| --- | --- | --- |
| `wireless/ieee80211_tgn` | IEEE 802.11-03/940r4 | A-F taps, overlap, ULA covariance, Bell Doppler, LoS, breakpoint loss, shadowing, fluorescent and vehicle options |
| `wireless/ieee80211_tgac` | IEEE 802.11-09/0308r12 | 20/40/80/160-MHz delay rules, up to 8x8, topology-bound uplink/downlink MU streams |
| `wireless/ieee80211_tgah` | IEEE 802.11-11/0968r4 | 1/2/4/8/16-MHz rules, 700-MHz-to-1-GHz use, AP-station/station-station deployment loss |
| `wireless/ieee80211_tgax` | IEEE 802.11-14/0882r4 | Indoor TGn/TGac composition; indoor penetration; outdoor UMi/UMa path loss and shadowing; fixed-window fast-object mode |

These are IEEE 802.11 task-group evaluation models, not 3GPP channel models.
The TGax source references ITU-R outdoor models. The comparison in
{doc}`tgax_itu_3gpp_comparison` records why the existing TR 38.901 resources
are not silently substituted.

## Review status and conformance boundary

All transcribed tables currently carry `pending_independent_review`. Checksums,
strict schemas, exact-row tests, physical sanity tests, and source provenance
are present, but this project does **not** claim independent verification or
formal IEEE conformance. TGax outdoor small-scale UMi/UMa LSP/cluster
calibration is not claimed in this release; the outdoor result uses the
explicitly selected A-F small-scale kernel plus the IEEE outdoor large-scale
and fast-object subset. This limitation is encoded in the capability matrix
and is not an alias for TR 38.901.

TGad and TGay are not implemented. Their 60-GHz directional and
quasi-deterministic physics require a separate millimetre-wave program.

## Choosing A-F

- A: optional one-tap stress case.
- B: residential/intra-room, 15-ns RMS delay spread.
- C: residential or small office, 30 ns.
- D: typical office, 50 ns.
- E: large office or warehouse, 100 ns.
- F: large indoor or pseudo-outdoor space, 150 ns.

The path axis of `CIRBatch` is the resolvable tap axis. Coefficients use
`[batch, rx, rx_antenna, tx, tx_antenna, path, time]`; delays use
`[batch, rx, tx, path]`.

## Sampling and replay

WLAN models use `WLANChannelSamplingRequest`, which composes the normal
`ChannelSamplingRequest` with an optional `TopologyBatch`:

```python
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211 import WLANChannelSamplingRequest
from channel_dataset.core import ExecutionContext

execution = ExecutionContext("numpy", "double", "cpu")
sampling = ChannelSamplingRequest(
    batch_size=4,
    num_time_samples=16,
    sampling_frequency_hz=20_000.0,
    execution=execution,
    random_stream=execution.random_stream(1234),
)
request = WLANChannelSamplingRequest(sampling)
cir = model.realize(request)
```

Large-scale loss, TGac multiuser operation, and all TGax scenarios require a
topology. A user stream is derived from endpoint coordinates for TGac MU
generation, so reordering users does not reassign their stochastic process.
Every other component has a distinct SHA-256-derived label.

Recreating the native backend stream with the same seed reproduces the same
window. NumPy and PyTorch each replay exactly within that backend; their random
samples need not match each other. Deterministic delays, covariance, loss, and
transforms are compared across backends.

## Time and large-scale behavior

The diffuse process uses the Bell spectrum from 03/940r4, truncated at five
times the environmental Doppler spread. Zero environmental speed gives a
constant diffuse coefficient over the returned window. Fluorescent and moving
object effects remain isolated optional components. A realization is
temporally consistent within its requested window; continuity between separate
calls is intentionally not promised.

The request sampling frequency must exceed the modeled Doppler bandwidth.
Fluorescent mode additionally requires more than 1 kHz. TGax fast-object mode
assigns 60 km/h to taps in clusters 2 and 3 and 3 km/h to the remainder.

Large-scale attenuation is enabled by `large_scale_mode` and applied once to
the normalized small-scale CIR. `path_loss_shadowing` adds an isolated
zero-mean Gaussian dB term. AWGN remains a separate noise component. TGax
indoor floor loss is nonlinear in floor count. Outdoor topology must provide
AP and station heights above 1 m and a 3-D distance of at least 10 m.

## Running the example

```powershell
python examples/wireless_ieee80211_tgn_ofdm.py
```

The example loads `configs/wireless/ieee80211/tgn_model_b.yaml`, constructs the
model through `ModelRegistry`, samples a seeded 2x2 CIR, converts it to OFDM,
and applies it to a resource grid.
