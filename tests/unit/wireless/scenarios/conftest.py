import pytest

from channel_dataset.channels.wireless import ScenarioConfig, TopologyBatch


@pytest.fixture
def topology_factory():
    def factory(backend, metadata, *, tx_height, rx_height, distance, indoor=False):
        tx = backend.array([[[0.0, 0.0, tx_height]]], dtype="float64")
        rx = backend.array([[[distance, 0.0, rx_height]]], dtype="float64")
        return TopologyBatch(
            tx_positions_m=tx,
            rx_positions_m=rx,
            tx_velocities_mps=backend.zeros((1, 1, 3), dtype="float64"),
            rx_velocities_mps=backend.zeros((1, 1, 3), dtype="float64"),
            tx_orientations_rad=backend.zeros((1, 1, 3), dtype="float64"),
            rx_orientations_rad=backend.zeros((1, 1, 3), dtype="float64"),
            rx_indoor=backend.full((1, 1), indoor, dtype="bool"),
            tx_site_ids=backend.zeros((1, 1), dtype="int64"),
            metadata=metadata,
        )
    return factory


@pytest.fixture
def scenario_config_factory():
    def factory(name):
        if name == "InH":
            return ScenarioConfig(name, 6.0e9, 120.0, 50.0, 3.0)
        if name == "InF":
            return ScenarioConfig(name, 28.0e9, 100.0, 50.0, 10.0, 0.4, 5.0)
        return ScenarioConfig(name, 3.5e9 if name != "RMa" else 2.0e9)
    return factory


@pytest.fixture
def valid_scenario_topology(topology_factory):
    def factory(name, backend, metadata, *, indoor=None, distance=None):
        settings = {
            "UMi": (10.0, 1.5, 100.0, False),
            "UMa": (25.0, 1.5, 100.0, False),
            "RMa": (35.0, 1.5, 1000.0, False),
            "InH": (3.0, 1.5, 20.0, True),
            "InF": (8.0, 1.5, 50.0, True),
        }
        tx_height, rx_height, default_distance, default_indoor = settings[name]
        return topology_factory(
            backend,
            metadata,
            tx_height=tx_height,
            rx_height=rx_height,
            distance=default_distance if distance is None else distance,
            indoor=default_indoor if indoor is None else indoor,
        )
    return factory
