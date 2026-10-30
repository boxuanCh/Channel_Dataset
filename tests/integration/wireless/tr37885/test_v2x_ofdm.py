from __future__ import annotations

import pytest

from channel_dataset.channels.wireless import (
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
)
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import ExecutionContext

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]


@pytest.mark.parametrize("backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)])
def test_scene_to_v2x_cir_to_ofdm_resource_grid(backend_name: str) -> None:
    pytest.importorskip(backend_name if backend_name == "torch" else "numpy")
    execution = ExecutionContext(backend_name, "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="A", num_vehicles=8)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(1, 2, execution, execution.random_stream(31))
    )
    sampling = ChannelSamplingRequest(1, 4, 15_000.0, execution, execution.random_stream(32))
    request = V2XChannelSamplingRequest(sampling, scene, 0, (0, 1), (2, 3))
    realization = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig()).realize(request)
    frequencies = subcarrier_frequencies(16, 15_000.0, execution)
    response = cir_to_ofdm(realization.cir, frequencies)
    backend = execution.array_backend
    signal = backend.ones((1, 2, 1, 4, 16), dtype=execution.complex_dtype)
    received = apply_ofdm(response, signal)

    assert response.shape == (1, 2, 1, 2, 1, 4, 16)
    assert backend.shape(received) == (1, 2, 1, 4, 16)
    assert backend.all_finite(received)
