from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811SystemChannel,
    TR38811SystemConfig,
    register_tr38811_models,
)
from channel_dataset.core import ExecutionContext
from channel_dataset.registry import ModelRegistry


def _config() -> TR38811SystemConfig:
    return TR38811SystemConfig(
        standard_version="15.4.0",
        deployment="D3",
        environment="urban",
        carrier_frequency_hz=2.0e9,
        bandwidth_hz=20.0e6,
        tx_array_profile="profiles/wireless/antenna_isotropic_siso.yaml",
        rx_array_profile="profiles/wireless/antenna_isotropic_siso.yaml",
        delay_spread_s=100e-9,
        desired_asa_degrees=15.0,
        desired_zsa_degrees=7.0,
        apply_large_scale=False,
    )


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(("los_value", "expected_second_delay"), [(False, 108.11e-9), (True, 0.0)])
def test_system_model_selects_los_or_nlos_kernel(
    los_value: bool, expected_second_delay: float, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    backend = get_backend("numpy")
    topology = zenith_topology_factory(execution)
    topology = replace(
        topology,
        los_mask=backend.full((1, 1, 1), los_value, dtype="bool", device="cpu"),
    )
    sampling = ChannelSamplingRequest(1, 3, 1000.0, execution, backend.random_stream(2))
    result = TR38811SystemChannel(_config()).realize(
        NTNChannelSamplingRequest(sampling, topology)
    )
    assert result.cir.shape == (1, 1, 1, 1, 1, 3, 3)
    assert np.isclose(result.cir.delays_s[0, 0, 0, 1], expected_second_delay)
    assert bool(result.los_mask[0, 0, 0]) is los_value


@pytest.mark.ntn
@pytest.mark.tr38811
def test_all_tr38811_models_are_registered() -> None:
    registry = ModelRegistry()
    register_tr38811_models(registry)
    assert registry.available_models() == (
        ("wireless", "tr38811_cdl"),
        ("wireless", "tr38811_flat"),
        ("wireless", "tr38811_system"),
        ("wireless", "tr38811_tdl"),
    )
