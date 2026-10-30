from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


@pytest.mark.statistical
def test_tdl_a_empirical_tap_powers_follow_profile(execution_context) -> None:
    model = TappedDelayLine(
        TDLConfig(
            profile="TDL-A", standard_version="16.1.0", delay_spread_s=100e-9,
            carrier_frequency_hz=3.5e9, bandwidth_hz=100e6, speed_mps=0.0,
            num_rx=1, num_rx_antennas=1, num_tx=1, num_tx_antennas=1,
            num_sinusoids=32, k_factor_db=None, correlation_mode="none",
            rx_correlation_coefficient=None, tx_correlation_coefficient=None,
            joint_correlation_matrix=None,
        )
    )
    request = ChannelSamplingRequest(
        5000, 1, 1000.0, execution_context, execution_context.random_stream(31415)
    )
    values = _as_numpy(model.realize(request).coefficients)[:, 0, 0, 0, 0, :, 0]
    empirical = np.mean(np.abs(values) ** 2, axis=0)
    expected = np.asarray(model.path_powers_linear)
    assert np.allclose(empirical, expected, rtol=0.12, atol=2e-4)
    assert np.mean(values[:, 0]).real == pytest.approx(0.0, abs=0.02)
