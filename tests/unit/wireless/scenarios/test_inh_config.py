import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario


def test_inh_requires_room_and_indoor_receiver(backend, realization_metadata, topology_factory) -> None:
    with pytest.raises(ValueError, match="room"):
        ScenarioConfig("InH", 6.0e9)
    scenario = create_scenario(ScenarioConfig("InH", 6.0e9, 120.0, 50.0, 3.0))
    valid = topology_factory(backend, realization_metadata, tx_height=3.0, rx_height=1.5, distance=20.0, indoor=True)
    assert scenario.validate_topology(valid).shape == (1, 1, 1)
    outdoor = topology_factory(backend, realization_metadata, tx_height=3.0, rx_height=1.5, distance=20.0, indoor=False)
    with pytest.raises(ValueError, match="indoor"):
        scenario.validate_topology(outdoor)
