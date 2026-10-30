from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811CDLChannel,
    TR38811CDLConfig,
    load_cdl_profile,
)
from channel_dataset.core import ExecutionContext


def _config() -> TR38811CDLConfig:
    return TR38811CDLConfig(
        profile="NTN-CDL-A",
        standard_version="15.4.0",
        deployment="D3",
        environment="urban",
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=20.0e6,
        delay_spread_s=100e-9,
        desired_asa_degrees=15.0,
        desired_zsa_degrees=7.0,
        tx_array_profile="profiles/wireless/antenna_isotropic_siso.yaml",
        rx_array_profile="profiles/wireless/antenna_isotropic_siso.yaml",
        apply_large_scale=False,
    )


@pytest.mark.ntn
@pytest.mark.tr38811
def test_cdl_a_resource_contains_standard_cluster_means() -> None:
    profile = load_cdl_profile("NTN-CDL-A")
    assert profile.table == "6.9.1-1"
    assert profile.delays == (0.0, 1.0811, 2.8416)
    assert profile.aoa_degrees == (178.8, -115.7, 111.5)
    assert profile.asd_degrees == 0.0


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_cdl_realization_shape_power_and_replay(
    backend_name: str, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    topology = zenith_topology_factory(execution)

    def realize():
        sampling = ChannelSamplingRequest(
            1, 4, 10_000.0, execution, backend.random_stream(99)
        )
        return TR38811CDLChannel(_config()).realize(
            NTNChannelSamplingRequest(sampling, topology)
        )

    first = realize()
    second = realize()
    first_values = (
        np.asarray(first.cir.coefficients)
        if backend_name == "numpy"
        else first.cir.coefficients.detach().numpy()
    )
    second_values = (
        np.asarray(second.cir.coefficients)
        if backend_name == "numpy"
        else second.cir.coefficients.detach().numpy()
    )
    assert first.cir.shape == (1, 1, 1, 1, 1, 3, 4)
    assert np.all(np.isfinite(first_values))
    assert np.array_equal(first_values, second_values)
