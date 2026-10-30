# 3GPP TR 37.885 urban V2X channels

The `tr37885` package implements the urban V2V evaluation scenario from 3GPP
TR 37.885 V15.3.0 at 6 GHz. It combines a reproducible reference street grid,
lane-constrained Options A/B traffic, LOS/NLOSv/NLOS link classification,
decomposed large-scale loss, and either stochastic system-level or fixed CDL
small-scale fading.

The numerical resources are versioned and currently marked
`pending_independent_review`. Tests verify their transcription checksums and
physical behavior, but this status is not a formal conformance claim.

## Choosing a model

| Registry key | Use |
| --- | --- |
| `wireless/tr37885_urban_system` | State-dependent stochastic delays, powers, angles, Doppler, and K-factor |
| `wireless/tr37885_urban_cdl` | Fixed standardized urban LOS, NLOSv, and NLOS CDL profiles |

Both return `V2XChannelRealization`. Its `cir` is an ordinary loss-scaled
`CIRBatch`, so `cir_to_ofdm`, `cir_to_time`, `apply_ofdm`, and `apply_time`
work without V2X-specific operators. Large-scale path loss, shadow fading, and
vehicle blockage remain available separately in `large_scale`.

The 6 GHz Option-1 vehicle antenna resource records the 3 dBi element limit,
half-wavelength spacing, 90-degree tilt, type-1/type-3 front/rear placement,
and type-2 rooftop layout from Tables 6.1.4-8 and 6.1.4-9. The coefficient
engine rotates location bearings by vehicle yaw; use
`load_vehicle_antenna_profile` to inspect the resolved placement.

## Scene and tensor contracts

`TR37885UrbanGrid()` constructs the 1299 m by 750 m reference area from
433 m by 250 m cells. Each street has four 3.5 m lanes and 3 m sidewalks.
`TR37885UrbanGridSimulator` places standardized vehicle types using exponential
bumper gaps and advances them at 100 ms intervals.

- trajectory values use `[batch, vehicle, mobility_time, xyz_or_ypr]`;
- link conditions use `[batch, rx, tx, mobility_time]`;
- channel coefficients use
  `[batch, rx, rx_antenna, tx, tx_antenna, path, waveform_time]`.

One channel realization represents one mobility snapshot and waveform window.
Use `realize_sequence` to advance through snapshots while carrying immutable
condition, shadowing, and episode state.

## Standard and controlled-variation modes

Standard mode locks the grid, 100 ms update interval, vehicle mixture, speeds,
turn probabilities, and finite-area exit behavior. Option A uses type-2
vehicles at 60 km/h with standardized intersection turns. Option B uses the
standardized type mixture and lane speeds, with the north-south fleet stopped.

Controlled-variation mode can change grid extent, traffic density and speed,
vehicle mixture, turns, or boundary behavior. Such runs set
`standard_baseline: false` in diagnostics and must not be reported as exact
standard evaluation cases. Real maps, traffic lights, lane changes, 3-D ray
tracing, and MAC/network simulation are outside this model.

## Minimal workflow

```python
from channel_dataset.channels.wireless import (
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import ExecutionContext

execution = ExecutionContext("numpy", "double", "cpu")
simulator = TR37885UrbanGridSimulator(
    traffic=TR37885TrafficConfig(option="A", num_vehicles=16)
)
scene = simulator.generate(
    V2XSceneSamplingRequest(1, 4, execution, execution.random_stream(37885))
)
sampling = ChannelSamplingRequest(
    1, 8, 15_000.0, execution, execution.random_stream(9)
)
request = V2XChannelSamplingRequest(sampling, scene, 0, (0,), (1,))
realization = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig()).realize(request)
frequencies = subcarrier_frequencies(64, 15_000.0, execution=execution)
response = cir_to_ofdm(realization.cir, frequencies)
```

For a complete executable scene-to-resource-grid pipeline, run
`python examples/wireless_tr37885_v2x_ofdm.py`.

## Reproducibility and persistence

Vehicle, lane, intersection, link, blockage, shadowing, cluster, and tap draws
use independently labelled child streams. A root seed exactly replays a run
within one backend; NumPy and PyTorch use native generators and are not
required to produce identical stochastic samples.

Use `save_v2x_realization` and `load_v2x_realization` for checksummed storage.
The saved form includes the CIR, topology, all link-condition and large-scale
arrays, carried state, diagnostics, configuration provenance, and standard
version, and it can be loaded into either supported backend.
