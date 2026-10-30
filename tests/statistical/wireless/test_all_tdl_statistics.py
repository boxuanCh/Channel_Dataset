from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig
from tests.statistical.statistical_assertions import as_numpy

pytestmark = pytest.mark.statistical


def _config(profile: str) -> TDLConfig:
    fixed = profile in {"TDLA30", "TDLB100", "TDLC300", "TDLC60"}
    carrier = 28.0e9 if profile == "TDLC60" else 3.5e9
    return TDLConfig(
        profile=profile,
        standard_version="16.1.0",
        delay_spread_s=None if fixed else 100e-9,
        carrier_frequency_hz=carrier,
        bandwidth_hz=100e6 if not fixed else 20e6,
        speed_mps=0.0,
        num_rx=1,
        num_rx_antennas=1,
        num_tx=1,
        num_tx_antennas=1,
        num_sinusoids=32,
        k_factor_db=None,
        correlation_mode="none",
        rx_correlation_coefficient=None,
        tx_correlation_coefficient=None,
        joint_correlation_matrix=None,
    )


@pytest.mark.parametrize(
    "profile", ["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E", "TDLA30", "TDLB100", "TDLC300", "TDLC60"]
)
def test_every_tdl_profile_has_correct_ensemble_tap_power_and_delay(profile: str) -> None:
    from channel_dataset.core import ExecutionContext

    execution = ExecutionContext("numpy", "double")
    model = TappedDelayLine(_config(profile))
    request = ChannelSamplingRequest(4_096, 1, 1_000.0, execution, execution.random_stream(71_000))
    cir = model.realize(request)
    coefficients = as_numpy(cir.coefficients)[:, 0, 0, 0, 0, :, 0]
    measured = np.mean(np.abs(coefficients) ** 2, axis=0)
    expected = np.asarray(model.path_powers_linear)
    # For each complex-Gaussian tap the relative 5-sigma mean error is 5/sqrt(n).
    tolerance = 5.0 / np.sqrt(request.batch_size)
    assert np.all(np.abs(measured - expected) <= tolerance * expected + 3.0e-4)
    assert np.allclose(as_numpy(cir.delays_s)[0, 0, 0], model.path_delays_s, atol=1.0e-18)
    assert np.isclose(np.sum(expected), 1.0, atol=1.0e-12)


@pytest.mark.parametrize("profile", ["TDL-A", "TDL-B", "TDL-C"])
def test_scalable_tdl_profile_resolves_requested_rms_delay_spread(profile: str) -> None:
    model = TappedDelayLine(_config(profile))
    delays = np.asarray(model.path_delays_s)
    powers = np.asarray(model.path_powers_linear)
    mean = np.sum(powers * delays)
    rms = np.sqrt(np.sum(powers * (delays - mean) ** 2))
    # The standardized normalized tables are rounded, so their normalized RMS
    # is close to, but not identically, one.
    assert rms == pytest.approx(100e-9, rel=0.035)


def test_tdl_diffuse_temporal_acf_matches_classical_jakes_spectrum() -> None:
    from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
    from channel_dataset.core import ExecutionContext

    execution = ExecutionContext("numpy", "double")
    maximum_doppler_hz = 100.0
    config = replace(
        _config("TDL-A"),
        speed_mps=maximum_doppler_hz * SPEED_OF_LIGHT_MPS / 3.5e9,
        num_sinusoids=64,
    )
    sampling_frequency_hz = 1_000.0
    request = ChannelSamplingRequest(
        128, 64, sampling_frequency_hz, execution, execution.random_stream(71_001)
    )
    values = as_numpy(TappedDelayLine(config).realize(request).coefficients)[
        :, 0, 0, 0, 0, 0, :
    ]
    lag = 2
    observed = np.mean(values[:, :-lag].conj() * values[:, lag:]) / np.mean(np.abs(values) ** 2)
    phase = 2.0 * np.pi * maximum_doppler_hz * lag / sampling_frequency_hz
    angles = np.linspace(0.0, 2.0 * np.pi, 200_000, endpoint=False)
    expected = np.mean(np.exp(1j * phase * np.cos(angles)))
    assert observed.real == pytest.approx(expected.real, abs=0.04)
    assert observed.imag == pytest.approx(0.0, abs=0.04)


def test_tdl_kronecker_spatial_covariance_matches_requested_coefficients() -> None:
    from channel_dataset.core import ExecutionContext

    execution = ExecutionContext("numpy", "double")
    config = replace(
        _config("TDL-A"),
        num_rx_antennas=2,
        num_tx_antennas=2,
        correlation_mode="kronecker",
        rx_correlation_coefficient=0.6,
        tx_correlation_coefficient=-0.3,
    )
    request = ChannelSamplingRequest(2_048, 1, 1.0, execution, execution.random_stream(71_002))
    values = as_numpy(TappedDelayLine(config).realize(request).coefficients)[
        :, 0, :, 0, :, 0, 0
    ].reshape(2_048, 4)
    covariance = values.conj().T @ values / values.shape[0]
    covariance /= np.mean(np.abs(values) ** 2)
    receive = np.array([[1.0, 0.6], [0.6, 1.0]])
    transmit = np.array([[1.0, -0.3], [-0.3, 1.0]])
    assert np.max(np.abs(covariance - np.kron(receive, transmit))) < 0.075
