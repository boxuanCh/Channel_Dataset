import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario


def test_umi_geometry_bounds(backend, realization_metadata, topology_factory) -> None:
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    topology = topology_factory(backend, realization_metadata, tx_height=10.0, rx_height=1.5, distance=100.0)
    assert scenario.validate_topology(topology).shape == (1, 1, 1)
    too_close = topology_factory(backend, realization_metadata, tx_height=10.0, rx_height=1.5, distance=9.0)
    with pytest.raises(ValueError, match="distance"):
        scenario.validate_topology(too_close)
