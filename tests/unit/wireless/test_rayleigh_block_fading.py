from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.fading import (
    RayleighBlockFading,
    RayleighBlockFadingConfig,
    parse_rayleigh_config,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _request(execution_context, seed: int = 11) -> ChannelSamplingRequest:
    return ChannelSamplingRequest(
        batch_size=256,
        num_time_samples=3,
        sampling_frequency_hz=1.0e6,
        execution=execution_context,
        random_stream=execution_context.random_stream(seed),
        request_id="test",
    )


def test_rayleigh_is_block_constant_and_replayable(execution_context) -> None:
    model = RayleighBlockFading(RayleighBlockFadingConfig(1, 2, 1, 2, 1.0))
    first = model.realize(_request(execution_context))
    second = model.realize(_request(execution_context))
    first_values = _as_numpy(first.coefficients)
    assert np.array_equal(first_values, _as_numpy(second.coefficients))
    assert np.array_equal(first_values[..., 0], first_values[..., 1])
    assert first.shape == (256, 1, 2, 1, 2, 1, 3)
    assert np.all(_as_numpy(first.delays_s) == 0.0)


def test_rayleigh_empirical_average_power(execution_context) -> None:
    model = RayleighBlockFading(RayleighBlockFadingConfig(1, 1, 1, 1, 2.0))
    values = _as_numpy(model.realize(_request(execution_context)).coefficients[..., 0])
    assert np.mean(np.abs(values) ** 2) == pytest.approx(2.0, rel=0.15)


def test_rayleigh_config_is_strict() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_rayleigh_config(
            {
                "num_rx": 1,
                "num_rx_antennas": 1,
                "num_tx": 1,
                "num_tx_antennas": 1,
                "average_power_linear": 1.0,
                "extra": 1,
            }
        )
