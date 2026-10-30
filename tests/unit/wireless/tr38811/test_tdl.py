from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811TDLChannel,
    TR38811TDLConfig,
)
from channel_dataset.core import ExecutionContext


def _config(profile: str = "NTN-TDL-A") -> TR38811TDLConfig:
    return TR38811TDLConfig(
        profile=profile,
        standard_version="15.4.0",
        deployment="D3",
        environment="rural",
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=20.0e6,
        delay_spread_s=100e-9,
        num_rx_antennas=1,
        num_tx_antennas=1,
        num_sinusoids=32,
        apply_large_scale=False,
    )


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_tdl_shape_replay_and_delay_scaling(
    backend_name: str, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    topology = zenith_topology_factory(execution)

    def realize():
        sampling = ChannelSamplingRequest(
            1, 8, 10_000.0, execution, backend.random_stream(1234), "ntn-test"
        )
        return TR38811TDLChannel(_config()).realize(
            NTNChannelSamplingRequest(sampling, topology)
        )

    first = realize()
    second = realize()
    assert first.cir.shape == (1, 1, 1, 1, 1, 3, 8)
    to_numpy = lambda value: np.asarray(value) if backend_name == "numpy" else value.detach().numpy()
    assert np.allclose(to_numpy(first.cir.delays_s), [0.0, 108.11e-9, 284.16e-9])
    assert np.array_equal(
        to_numpy(first.cir.coefficients), to_numpy(second.cir.coefficients)
    )


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.statistical
def test_tdl_a_tap_power_follows_standard_profile(zenith_topology_factory) -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    backend = get_backend("numpy")
    batch_size = 4000
    sampling = ChannelSamplingRequest(
        batch_size, 1, 10_000.0, execution, backend.random_stream(88)
    )
    result = TR38811TDLChannel(_config()).realize(
        NTNChannelSamplingRequest(
            sampling, zenith_topology_factory(execution, batch_size=batch_size)
        )
    )
    measured = np.mean(np.abs(result.cir.coefficients[:, 0, 0, 0, 0, :, 0]) ** 2, axis=0)
    expected = np.power(10.0, np.array([0.0, -4.675, -6.482]) / 10.0)
    expected /= expected.sum()
    assert np.allclose(measured, expected, rtol=0.08, atol=0.01)
