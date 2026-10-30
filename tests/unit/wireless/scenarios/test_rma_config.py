import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario


def test_rma_frequency_and_geometry(backend, realization_metadata, topology_factory) -> None:
    with pytest.raises(ValueError, match="frequency"):
        create_scenario(ScenarioConfig("RMa", 40.0e9))
    scenario = create_scenario(ScenarioConfig("RMa", 2.0e9))
    topology = topology_factory(backend, realization_metadata, tx_height=35.0, rx_height=1.5, distance=1000.0)
    assert scenario.validate_topology(topology).shape == (1, 1, 1)
