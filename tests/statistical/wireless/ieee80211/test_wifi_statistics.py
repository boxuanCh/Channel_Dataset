from __future__ import annotations

from math import exp, pi

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211 import (
    TGnChannel,
    TGnConfig,
    WLANChannelSamplingRequest,
    load_ieee80211_profile,
)
from channel_dataset.channels.wireless.ieee80211.temporal import bell_doppler_process
from channel_dataset.core import ExecutionContext


def _config(profile: str, **overrides: object) -> TGnConfig:
    values: dict[str, object] = {
        "profile": profile,
        "standard_version": "03/940r4",
        "carrier_frequency_hz": 5.25e9,
        "bandwidth_hz": 20e6,
        "num_rx": 1,
        "num_rx_antennas": 1,
        "num_tx": 1,
        "num_tx_antennas": 1,
        "environmental_speed_mps": 0.0,
        "num_sinusoids": 32,
        "spatial_quadrature_order": 128,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5,
        "line_of_sight": False,
        "large_scale_mode": "none",
        "fluorescent_effect": False,
        "moving_vehicle_effect": False,
        "normalize_profile_power": True,
    }
    values.update(overrides)
    return TGnConfig(**values)


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_tgn_los_first_tap_has_configured_rician_k_factor() -> None:
    execution = ExecutionContext("numpy", "double")
    batch_size = 1_024
    request = WLANChannelSamplingRequest(
        ChannelSamplingRequest(
            batch_size, 1, 1e3, execution, execution.random_stream(810)
        )
    )
    samples = TGnChannel(_config("D", line_of_sight=True)).realize(request)
    first_tap = samples.coefficients[:, 0, 0, 0, 0, 0, 0]
    normalized_power = np.abs(first_tap) ** 2
    observed_fourth_moment = float(
        np.mean(normalized_power**2) / np.mean(normalized_power) ** 2
    )
    k_linear = 10.0 ** (load_ieee80211_profile("D").los_k_factor_db / 10.0)
    expected_fourth_moment = (k_linear**2 + 4.0 * k_linear + 2.0) / (k_linear + 1.0) ** 2
    assert observed_fourth_moment == pytest.approx(expected_fourth_moment, abs=0.09)


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_tgn_empirical_spatial_covariance_matches_profile() -> None:
    execution = ExecutionContext("numpy", "double")
    request = WLANChannelSamplingRequest(
        ChannelSamplingRequest(2_048, 1, 1e3, execution, execution.random_stream(811))
    )
    model = TGnChannel(_config("A", num_rx_antennas=2))
    cir = model.realize(request)
    samples = cir.coefficients[:, 0, :, 0, 0, 0, 0]
    empirical = samples.T @ samples.conj() / samples.shape[0]
    empirical /= np.sqrt(np.outer(np.diag(empirical), np.diag(empirical)))
    expected = model.diagnostics(request).receive_covariances[0]
    assert np.allclose(empirical, expected, atol=0.06)


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.statistical
def test_bell_process_autocorrelation_matches_discrete_spectrum() -> None:
    execution = ExecutionContext("numpy", "double")
    carrier_frequency_hz = 5.25e9
    speed_mps = 0.3
    sampling_frequency_hz = 1_000.0
    num_sinusoids = 64
    lag = 10
    values = bell_doppler_process(
        execution.random_stream(812),
        execution.array_backend,
        (256,),
        256,
        sampling_frequency_hz,
        carrier_frequency_hz,
        speed_mps,
        num_sinusoids=num_sinusoids,
        float_dtype="float64",
        complex_dtype="complex128",
        device="cpu",
    )
    observed = np.mean(values[:, :-lag].conj() * values[:, lag:]) / np.mean(np.abs(values) ** 2)
    maximum_doppler_hz = speed_mps * carrier_frequency_hz / 299_792_458.0
    frequencies = np.linspace(
        -5.0 * maximum_doppler_hz,
        5.0 * maximum_doppler_hz,
        num_sinusoids,
        endpoint=False,
    ) + 5.0 * maximum_doppler_hz / num_sinusoids
    powers = 1.0 / (1.0 + 9.0 * (frequencies / maximum_doppler_hz) ** 2)
    expected = np.sum(powers * np.exp(1j * 2.0 * pi * frequencies * lag / sampling_frequency_hz)) / np.sum(powers)
    assert observed.real == pytest.approx(expected.real, abs=0.025)
    assert observed.imag == pytest.approx(expected.imag, abs=0.025)
    untruncated_limit = exp(
        -2.0 * pi * maximum_doppler_hz * lag / (3.0 * sampling_frequency_hz)
    )
    assert observed.real == pytest.approx(untruncated_limit, abs=0.07)
