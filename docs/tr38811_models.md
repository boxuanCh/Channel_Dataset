# 3GPP TR 38.811 non-terrestrial channels

The `channel_dataset.channels.wireless.tr38811` package models fixed-window
service links for GEO, non-GEO satellite, and HAPS deployments using the
TR 38.811 V15.4.0 baseline. It uses caller-supplied ECEF state vectors, so an
orbit propagator can remain authoritative and no TLE/SGP4 dependency is
required.

## Model selection

| Registry key | Use |
| --- | --- |
| `wireless/tr38811_flat` | Narrowband two-state land-mobile-satellite fading with explicit Loo parameters |
| `wireless/tr38811_tdl` | NTN-TDL-A/B/C/D frequency-selective fading |
| `wireless/tr38811_cdl` | Antenna-aware NTN-CDL-A/B/C/D fading |
| `wireless/tr38811_system` | Per-link LOS selection and standardized NTN CDL composition |

All models return `NTNChannelRealization`. Its `cir.delays_s` contains only
excess multipath delays. Absolute slant-range delay is recorded separately in
`geometry.propagation_delay_s`; this avoids accidentally allocating an
impractically large time-domain lag window.

## Topology and geometry

`NTNTopologyBatch` positions, velocities, and orientations use
`[batch, endpoint, trajectory_time, xyz]` ECEF arrays. Trajectory time is one
or the request's number of samples. Endpoint kinds are `ground`, `satellite`,
or `haps`. Optional LOS and link masks use `[batch, rx, tx]`.

Derived scalar geometry uses `[batch, rx, tx, time]`: range, elevation,
azimuth, central angle, propagation delay, radial velocity, common Doppler,
and visibility. `direction_rx_to_tx` appends an xyz axis. Positive radial
velocity is opening speed, so common Doppler follows
`f_D = -v_radial f_c / c`.

```python
from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    NTNChannelSamplingRequest,
    TR38811TDLChannel,
    TR38811TDLConfig,
)

sampling = ChannelSamplingRequest(
    batch_size=1,
    num_time_samples=32,
    sampling_frequency_hz=15_360_000.0,
    execution=execution,
    random_stream=execution.random_stream(2026),
)
request = NTNChannelSamplingRequest(sampling=sampling, topology=topology)
channel = TR38811TDLChannel(TR38811TDLConfig(
    profile="NTN-TDL-A",
    standard_version="15.4.0",
    deployment="D3",
    environment="rural",
    carrier_frequency_hz=2.0e9,
    bandwidth_hz=20.0e6,
    delay_spread_s=100.0e-9,
    num_rx_antennas=1,
    num_tx_antennas=1,
))
realization = channel.realize(request)
```

Use `geodetic_to_ecef`, `geostationary_state`,
`circular_equatorial_orbit`, or `constant_velocity_trajectory` to build simple
state vectors. The circular-orbit helper intentionally omits Earth rotation;
production callers should pass state vectors from their selected propagator.

## Loss and atmosphere

`NTNLargeScaleComponents` retains free-space, clutter, shadow, building-entry,
gas, rain, cloud, and scintillation attenuation separately, plus their total.
Effects can be disabled when an external link budget already contains them.
LOS is sampled once per link and held over the requested window.

Site-specific meteorology is represented by `NTNAtmosphereConfig` or an
`NTNAtmosphereProvider`. The current module provides elevation mapping and
clear separation of components, but does not ship global maps or claim a full
P.676/P.618/P.840 recommendation engine. Likewise, the flat model requires
the applicable P.681 Loo triplets explicitly rather than guessing a site
category.

## OFDM timing modes

`ntn_to_ofdm(realization, frequencies_hz)` produces a synchronized response.
Set `include_propagation_delay=True` to multiply by
`exp(-j 2 pi f tau(t))`. Absolute-delay time-domain application is rejected;
the ordinary `cir_to_time` transform continues to operate on excess delays.

## Replay and persistence

Random effects use labeled child streams for state, LOS, shadowing, direct
phase and magnitude, diffuse rays, taps, and links. Recreating a native
backend stream with the same seed reproduces the same window. NumPy and
PyTorch stochastic samples need not be identical.

Use `save_ntn_realization` and `load_ntn_realization` for the checksummed
schema-version-1 directory format. CIR, geometry, large-scale components, LOS
mask, and selected diagnostics are preserved and can be loaded into either
backend.

## Validation and standards status

Tests cover exact packaged resource rows, geometry and Doppler identities,
loss equations, delay and OFDM phase, fixed-state and power statistics,
spatial array response, Faraday power preservation, replay, gradients,
corruption detection, and backend parity. Resources cite the official
`38811-f40` archive and currently carry `pending_independent_review`; the
package therefore does not claim formal or independently verified 3GPP/ITU
conformance.

Run the end-to-end example with:

```powershell
python examples/wireless_tr38811_ntn_ofdm.py
```
