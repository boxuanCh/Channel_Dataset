import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario


def test_uma_height_bounds(backend, realization_metadata, topology_factory) -> None:
    scenario = create_scenario(ScenarioConfig("UMa", 3.5e9))
    valid = topology_factory(backend, realization_metadata, tx_height=25.0, rx_height=10.0, distance=100.0)
    assert scenario.validate_topology(valid).shape == (1, 1, 1)
    invalid = topology_factory(backend, realization_metadata, tx_height=24.0, rx_height=10.0, distance=100.0)
    with pytest.raises(ValueError, match="BS height"):
        scenario.validate_topology(invalid)
