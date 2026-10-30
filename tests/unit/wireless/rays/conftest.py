import pytest

from channel_dataset.channels.wireless import (
    ScenarioConfig,
    ScenarioRayGenerator,
    TopologyBatch,
    create_scenario,
)


@pytest.fixture
def ray_case(backend, realization_metadata):
    zero = backend.zeros((1, 1, 3), dtype="float64")
    topology = TopologyBatch(
        backend.array([[[0.0, 0.0, 10.0]]], dtype="float64"),
        backend.array([[[100.0, 0.0, 1.5]]], dtype="float64"),
        zero,
        zero,
        zero,
        zero,
        backend.zeros((1, 1), dtype="bool"),
        backend.zeros((1, 1), dtype="int64"),
        realization_metadata,
        los_mask=backend.ones((1, 1, 1), dtype="bool"),
    )
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    large_scale = scenario.sample_large_scale(topology, backend.random_stream(100))
    generator = ScenarioRayGenerator("UMi")
    rays = generator.generate(topology, large_scale, backend.random_stream(200))
    return topology, large_scale, generator, rays
