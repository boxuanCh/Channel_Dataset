from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211 import (
    TGnChannel,
    TGnConfig,
    WLANChannelSamplingRequest,
    load_ieee80211_profile,
)
from channel_dataset.channels.wireless.ieee80211.temporal import (
    apply_fluorescent_effect,
    apply_moving_vehicle_effect,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _config(profile: str = "B", **overrides) -> TGnConfig:
    values = {
        "profile": profile, "standard_version": "03/940r4",
        "carrier_frequency_hz": 5.25e9, "bandwidth_hz": 20e6,
        "num_rx": 1, "num_rx_antennas": 1, "num_tx": 1,
        "num_tx_antennas": 1, "environmental_speed_mps": 0.0,
        "num_sinusoids": 32, "spatial_quadrature_order": 64,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5, "line_of_sight": False,
        "large_scale_mode": "none", "fluorescent_effect": False,
        "moving_vehicle_effect": False, "normalize_profile_power": True,
    }
    values.update(overrides)
    return TGnConfig(**values)


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_static_cir_contract_and_replay(execution_context) -> None:
    sampling = ChannelSamplingRequest(
        2, 8, 1e3, execution_context, execution_context.random_stream(901)
    )
    model = TGnChannel(_config())
    first = model.realize(WLANChannelSamplingRequest(sampling))
    replay_sampling = ChannelSamplingRequest(
        2, 8, 1e3, execution_context, execution_context.random_stream(901)
    )
    second = model.realize(WLANChannelSamplingRequest(replay_sampling))
    assert first.shape == (2, 1, 1, 1, 1, 9, 8)
    assert np.array_equal(_as_numpy(first.coefficients), _as_numpy(second.coefficients))
    assert np.all(_as_numpy(first.coefficients)[..., :1] == _as_numpy(first.coefficients))
    assert first.metadata.standard_version == "03/940r4"
    assert first.metadata.root_seed == 901


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_tgn_empirical_tap_powers_follow_profile(execution_context) -> None:
    sampling = ChannelSamplingRequest(
        1024, 1, 1e3, execution_context, execution_context.random_stream(333)
    )
    cir = TGnChannel(_config()).realize(WLANChannelSamplingRequest(sampling))
    observed = np.mean(np.abs(_as_numpy(cir.coefficients)) ** 2, axis=(0, 1, 2, 3, 4, 6))
    expected = np.array([tap.power_linear for tap in load_ieee80211_profile("B").taps])
    assert np.allclose(observed, expected, rtol=0.16, atol=0.003), (observed, expected)


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_diagnostics_do_not_perturb_draws(execution_context) -> None:
    model = TGnChannel(_config(num_rx_antennas=2, num_tx_antennas=2))
    sampling = ChannelSamplingRequest(
        1, 3, 1e3, execution_context, execution_context.random_stream(19)
    )
    request = WLANChannelSamplingRequest(sampling)
    model.diagnostics(request)
    after = model.realize(request)
    fresh = model.realize(
        WLANChannelSamplingRequest(
            ChannelSamplingRequest(1, 3, 1e3, execution_context, execution_context.random_stream(19))
        )
    )
    assert np.array_equal(_as_numpy(after.coefficients), _as_numpy(fresh.coefficients))


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_time_variation_and_aliasing_validation(execution_context) -> None:
    config = _config(profile="A", environmental_speed_mps=1.0)
    sampling = ChannelSamplingRequest(
        1, 16, 1e3, execution_context, execution_context.random_stream(12)
    )
    cir = TGnChannel(config).realize(WLANChannelSamplingRequest(sampling))
    values = _as_numpy(cir.coefficients).reshape(-1)
    assert not np.allclose(values, values[0])
    bad = ChannelSamplingRequest(1, 2, 10.0, execution_context, execution_context.random_stream(12))
    with pytest.raises(ValueError, match="Doppler bandwidth"):
        TGnChannel(config).realize(WLANChannelSamplingRequest(bad))


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_fluorescent_effect_has_documented_spectral_lines(backend, execution_context) -> None:
    sampling_frequency_hz = 4_000.0
    samples = 4_000
    values = backend.ones((samples,), dtype="complex128")
    result = apply_fluorescent_effect(
        values,
        execution_context.random_stream(200),
        backend,
        sampling_frequency_hz,
        float_dtype="float64",
        device="cpu",
    )
    spectrum = np.abs(np.fft.fft(_as_numpy(result) - 1.0))
    frequencies = np.fft.fftfreq(samples, 1.0 / sampling_frequency_hz)
    for expected_hz in (100.0, 300.0, 500.0):
        index = int(np.argmin(np.abs(frequencies - expected_hz)))
        assert spectrum[index] > 100.0 * np.median(spectrum)


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_moving_vehicle_effect_is_centered_at_40_kmh_doppler(backend, execution_context) -> None:
    sampling_frequency_hz = 2_000.0
    carrier_frequency_hz = 5.25e9
    samples = 8_192
    values = backend.zeros((samples,), dtype="complex128")
    first = apply_moving_vehicle_effect(
        values,
        execution_context.random_stream(201),
        backend,
        sampling_frequency_hz,
        carrier_frequency_hz,
        float_dtype="float64",
        device="cpu",
    )
    replay = apply_moving_vehicle_effect(
        values,
        execution_context.random_stream(201),
        backend,
        sampling_frequency_hz,
        carrier_frequency_hz,
        float_dtype="float64",
        device="cpu",
    )
    first_array = _as_numpy(first)
    assert np.array_equal(first_array, _as_numpy(replay))
    spectrum = np.abs(np.fft.fft(first_array)) ** 2
    frequencies = np.fft.fftfreq(samples, 1.0 / sampling_frequency_hz)
    positive = frequencies > 0
    centroid_hz = float(np.sum(frequencies[positive] * spectrum[positive]) / np.sum(spectrum[positive]))
    expected_hz = (40.0 / 3.6) * carrier_frequency_hz / 299_792_458.0
    assert centroid_hz == pytest.approx(expected_hz, rel=0.03)
