import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario


def test_inf_factory_constraints(backend, realization_metadata, topology_factory) -> None:
    with pytest.raises(ValueError, match="clutter_density"):
        ScenarioConfig("InF", 28.0e9, 100.0, 50.0, 10.0)
    scenario = create_scenario(ScenarioConfig("InF", 28.0e9, 100.0, 50.0, 10.0, 0.4, 5.0))
    topology = topology_factory(backend, realization_metadata, tx_height=8.0, rx_height=1.5, distance=50.0, indoor=True)
    assert scenario.validate_topology(topology).shape == (1, 1, 1)
