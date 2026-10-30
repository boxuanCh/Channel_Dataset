import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    ScenarioConfig,
    ScenarioRayGenerator,
    SystemLevelWirelessChannel,
    TopologyBatch,
    create_scenario,
)
from channel_dataset.channels.wireless.antennas import load_antenna_layout
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies


@pytest.mark.parametrize(
    ("name", "configuration", "tx_height", "rx_height", "distance", "indoor"),
    [
        ("RMa", ScenarioConfig("RMa", 2.0e9), 35.0, 1.5, 1000.0, False),
        ("UMa", ScenarioConfig("UMa", 3.5e9), 25.0, 1.5, 100.0, False),
        ("UMi", ScenarioConfig("UMi", 3.5e9), 10.0, 1.5, 100.0, False),
        ("InH", ScenarioConfig("InH", 6.0e9, 120.0, 50.0, 3.0), 3.0, 1.5, 20.0, True),
        ("InF", ScenarioConfig("InF", 28.0e9, 100.0, 50.0, 10.0, 0.4, 5.0), 8.0, 1.5, 50.0, True),
    ],
)
def test_supported_scenario_produces_valid_ray_batch(
    backend,
    realization_metadata,
    execution_context,
    name,
    configuration,
    tx_height,
    rx_height,
    distance,
    indoor,
):
    zero = backend.zeros((1, 1, 3), dtype="float64")
    topology = TopologyBatch(
        backend.array([[[0.0, 0.0, tx_height]]], dtype="float64"),
        backend.array([[[distance, 0.0, rx_height]]], dtype="float64"),
        zero,
        zero,
        zero,
        zero,
        backend.full((1, 1), indoor, dtype="bool"),
        backend.zeros((1, 1), dtype="int64"),
        realization_metadata,
        los_mask=backend.ones((1, 1, 1), dtype="bool"),
    )
    scenario = create_scenario(configuration)
    large_scale = scenario.sample_large_scale(topology, backend.random_stream(3))
    rays = ScenarioRayGenerator(name).generate(
        topology, large_scale, backend.random_stream(4)
    )
    assert rays.shape[:3] == topology.shape
    assert rays.shape[-1] == 20
    assert abs(float(backend.to_scalar(backend.sum(rays.powers_linear))) - 1.0) < 1e-10
    layout = load_antenna_layout("profiles/wireless/antenna_isotropic_siso.yaml")
    model = SystemLevelWirelessChannel(configuration, topology, layout, layout)
    request = ChannelSamplingRequest(
        1, 2, 1000.0, execution_context, execution_context.random_stream(10)
    )
    cir = model.realize(request)
    frequencies = subcarrier_frequencies(8, 15_000.0, execution_context)
    response = cir_to_ofdm(cir, frequencies)
    assert response.shape == (1, 1, 1, 1, 1, 2, 8)
