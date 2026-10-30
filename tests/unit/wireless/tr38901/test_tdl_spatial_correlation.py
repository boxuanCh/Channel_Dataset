from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _sample(execution_context, mode, **correlation):
    model = TappedDelayLine(
        TDLConfig(
            profile="TDL-A", standard_version="16.1.0", delay_spread_s=100e-9,
            carrier_frequency_hz=3.5e9, bandwidth_hz=100e6, speed_mps=0.0,
            num_rx=1, num_rx_antennas=2, num_tx=1, num_tx_antennas=1,
            num_sinusoids=32, k_factor_db=None, correlation_mode=mode,
            rx_correlation_coefficient=correlation.get("rx"),
            tx_correlation_coefficient=correlation.get("tx"),
            joint_correlation_matrix=correlation.get("joint"),
        )
    )
    request = ChannelSamplingRequest(
        3000, 1, 1000.0, execution_context, execution_context.random_stream(91)
    )
    return _as_numpy(model.realize(request).coefficients)[:, 0, :, 0, 0, 0, 0]


def _empirical_correlation(values):
    return np.mean(values[:, 0] * np.conj(values[:, 1])) / np.sqrt(
        np.mean(np.abs(values[:, 0]) ** 2) * np.mean(np.abs(values[:, 1]) ** 2)
    )


def test_kronecker_receive_correlation(execution_context) -> None:
    values = _sample(execution_context, "kronecker", rx=0.7, tx=0.0)
    assert _empirical_correlation(values).real == pytest.approx(0.7, abs=0.06)


def test_joint_spatial_correlation(execution_context) -> None:
    values = _sample(execution_context, "joint", joint=((1.0, 0.5), (0.5, 1.0)))
    assert _empirical_correlation(values).real == pytest.approx(0.5, abs=0.06)
