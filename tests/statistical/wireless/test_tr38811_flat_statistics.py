from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811FlatFadingChannel,
    TR38811FlatFadingConfig,
)
from channel_dataset.channels.wireless.tr38811.geometry import EARTH_RADIUS_M
from channel_dataset.channels.wireless.tr38811.types import NTNTopologyBatch
from channel_dataset.core import ExecutionContext, RealizationMetadata


def _topology(execution: ExecutionContext, batch: int) -> NTNTopologyBatch:
    backend = get_backend(execution.backend)
    tx = backend.array(
        [[[[EARTH_RADIUS_M + 600_000.0, 0.0, 0.0]]]] * batch,
        dtype=execution.float_dtype,
        device=execution.device,
    )
    rx = backend.array(
        [[[[EARTH_RADIUS_M, 0.0, 0.0]]]] * batch,
        dtype=execution.float_dtype,
        device=execution.device,
    )
    zero = backend.zeros((batch, 1, 1, 3), dtype=execution.float_dtype, device=execution.device)
    indoor = backend.zeros((batch, 1), dtype="bool", device=execution.device)
    metadata = RealizationMetadata(
        "wireless", "topology", 1, execution.backend, execution.precision,
        execution.device, "test", "0" * 64, {}, "0.1.0",
    )
    return NTNTopologyBatch(
        tx, rx, zero, zero, zero, zero, "satellite", "ground", indoor, metadata
    )


def _config(probability: float, mode: str = "sample") -> TR38811FlatFadingConfig:
    return TR38811FlatFadingConfig(
        standard_version="15.4.0",
        deployment="D3",
        environment="rural",
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=5.0e6,
        good_state_probability=probability,
        good_direct_mean_db=0.0,
        good_direct_std_db=0.0,
        good_multipath_power_db=-300.0,
        bad_direct_mean_db=-300.0,
        bad_direct_std_db=0.0,
        bad_multipath_power_db=0.0,
        state_mode=mode,
        num_sinusoids=32,
        apply_large_scale=False,
    )


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.statistical
def test_flat_state_probability_and_conditioned_power() -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    backend = get_backend("numpy")
    batch = 6000
    sampling = ChannelSamplingRequest(batch, 1, 1000.0, execution, backend.random_stream(741))
    result = TR38811FlatFadingChannel(_config(0.7)).realize(
        NTNChannelSamplingRequest(sampling, _topology(execution, batch))
    )
    good = np.asarray(result.diagnostics.values["good_state"])
    power = np.abs(np.asarray(result.cir.coefficients)[:, 0, 0, 0, 0, 0, 0]) ** 2
    assert abs(float(np.mean(good)) - 0.7) < 0.025
    assert np.isclose(float(np.mean(power[good[:, 0, 0]])), 1.0, atol=0.03)
    assert np.isclose(float(np.mean(power[~good[:, 0, 0]])), 1.0, rtol=0.08)


@pytest.mark.ntn
@pytest.mark.tr38811
def test_flat_forced_good_state_is_replayable() -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    backend = get_backend("numpy")
    topology = _topology(execution, 2)

    def realize():
        sampling = ChannelSamplingRequest(2, 8, 1000.0, execution, backend.random_stream(4))
        return TR38811FlatFadingChannel(_config(0.5, "good")).realize(
            NTNChannelSamplingRequest(sampling, topology)
        )

    assert np.array_equal(realize().cir.coefficients, realize().cir.coefficients)
