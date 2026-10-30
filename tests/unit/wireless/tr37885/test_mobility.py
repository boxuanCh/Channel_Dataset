from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.tr37885 import (
    TR37885TrafficConfig,
    TR37885UrbanGridSimulator,
    V2XSceneSamplingRequest,
)
from channel_dataset.core import ExecutionContext

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885, pytest.mark.mobility]


@pytest.mark.parametrize("backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)])
def test_option_a_scene_shape_types_speed_and_replay(backend_name):
    pytest.importorskip(backend_name if backend_name == "torch" else "numpy")
    execution = ExecutionContext(backend_name, "double", "cpu")
    request = V2XSceneSamplingRequest(2, 4, execution, execution.random_stream(123))
    simulator = TR37885UrbanGridSimulator(traffic=TR37885TrafficConfig(num_vehicles=8))
    first = simulator.generate(request)
    second = simulator.generate(
        V2XSceneSamplingRequest(2, 4, execution, execution.random_stream(123))
    )
    backend = execution.array_backend
    assert first.shape == (2, 8, 4)
    assert backend.to_scalar(backend.min(first.vehicle_type_codes)) == 2
    assert backend.to_scalar(backend.max(first.vehicle_type_codes)) == 2
    assert backend.to_scalar(backend.max(backend.abs(first.positions_m - second.positions_m))) == 0.0


def test_option_b_vertical_vehicles_are_stationary():
    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="B", num_vehicles=32)
    )
    scene = simulator.generate(V2XSceneSamplingRequest(1, 2, execution, execution.random_stream(9)))
    backend = execution.array_backend
    for vehicle in range(scene.shape[1]):
        lane = int(backend.to_scalar(scene.lane_ids[0, vehicle, 0]))
        if simulator.grid.lane(lane).orientation == "vertical":
            speed = backend.to_scalar(backend.sum(scene.velocities_mps[0, vehicle, 0] ** 2))
            assert speed == 0.0
            assert bool(backend.to_scalar(scene.active_mask[0, vehicle, 1]))
