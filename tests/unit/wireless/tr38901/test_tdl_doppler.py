from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _model(speed_mps: float, profile="TDL-A") -> TappedDelayLine:
    return TappedDelayLine(
        TDLConfig(
            profile=profile, standard_version="16.1.0", delay_spread_s=100e-9,
            carrier_frequency_hz=3.5e9, bandwidth_hz=100e6, speed_mps=speed_mps,
            num_rx=1, num_rx_antennas=1, num_tx=1, num_tx_antennas=1,
            num_sinusoids=32, k_factor_db=None, correlation_mode="none",
            rx_correlation_coefficient=None, tx_correlation_coefficient=None,
            joint_correlation_matrix=None,
        )
    )


def _sample(model, execution_context, samples=32):
    request = ChannelSamplingRequest(
        2, samples, 2000.0, execution_context, execution_context.random_stream(42)
    )
    return _as_numpy(model.realize(request).coefficients)


def test_speed_to_doppler_conversion() -> None:
    model = _model(12.0)
    assert model.maximum_doppler_hz == pytest.approx(12.0 * 3.5e9 / SPEED_OF_LIGHT_MPS)


def test_zero_speed_coefficients_are_constant(execution_context) -> None:
    values = _sample(_model(0.0), execution_context)
    assert np.allclose(values, values[..., :1], atol=1e-15)


def test_nonzero_speed_changes_coefficients(execution_context) -> None:
    values = _sample(_model(30.0), execution_context)
    assert not np.allclose(values[..., 0], values[..., -1])


def test_los_component_has_point_doppler(execution_context) -> None:
    model = _model(30.0, profile="TDL-D")
    values = _sample(model, execution_context, samples=4)
    los_index = model.profile.component_types.index("los")
    los = values[0, 0, 0, 0, 0, los_index]
    phase_steps = los[1:] / los[:-1]
    expected = np.exp(1j * 2 * np.pi * 0.7 * model.maximum_doppler_hz / 2000.0)
    assert np.allclose(phase_steps, expected, atol=1e-10)
