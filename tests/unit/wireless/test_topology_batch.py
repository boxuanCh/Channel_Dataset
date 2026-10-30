import numpy as np
import pytest

from channel_dataset.channels.wireless import TopologyBatch


def _topology(backend, metadata, *, rx_count: int = 3) -> TopologyBatch:
    tx_shape = (2, 2, 3)
    rx_shape = (2, rx_count, 3)
    return TopologyBatch(
        tx_positions_m=backend.zeros(tx_shape, dtype="float64"),
        rx_positions_m=backend.ones(rx_shape, dtype="float64"),
        tx_velocities_mps=backend.zeros(tx_shape, dtype="float64"),
        rx_velocities_mps=backend.zeros(rx_shape, dtype="float64"),
        tx_orientations_rad=backend.zeros(tx_shape, dtype="float64"),
        rx_orientations_rad=backend.zeros(rx_shape, dtype="float64"),
        rx_indoor=backend.zeros((2, rx_count), dtype="bool"),
        tx_site_ids=backend.array([[0, 1], [0, 1]], dtype="int64"),
        metadata=metadata,
        los_mask=backend.ones((2, rx_count, 2), dtype="bool"),
        link_mask=backend.ones((2, rx_count, 2), dtype="bool"),
    )


def test_topology_validates_and_selects_views(backend, realization_metadata) -> None:
    topology = _topology(backend, realization_metadata)
    selected = topology.select_link(1, 2, 1)
    assert topology.shape == (2, 3, 2)
    assert selected.shape == (1, 1, 1)
    if type(topology.tx_positions_m).__module__.partition(".")[0] == "torch":
        assert (
            topology.tx_positions_m.untyped_storage().data_ptr()
            == selected.tx_positions_m.untyped_storage().data_ptr()
        )
    else:
        assert np.shares_memory(topology.tx_positions_m, selected.tx_positions_m)


def test_topology_rejects_endpoint_shape_mismatch(backend, realization_metadata) -> None:
    topology = _topology(backend, realization_metadata)
    with pytest.raises(ValueError, match="rx_velocities_mps"):
        TopologyBatch(
            topology.tx_positions_m,
            topology.rx_positions_m,
            topology.tx_velocities_mps,
            backend.zeros((2, 2, 3), dtype="float64"),
            topology.tx_orientations_rad,
            topology.rx_orientations_rad,
            topology.rx_indoor,
            topology.tx_site_ids,
            realization_metadata,
        )


def test_topology_rejects_invalid_link_selection(backend, realization_metadata) -> None:
    topology = _topology(backend, realization_metadata)
    with pytest.raises(IndexError, match="rx"):
        topology.select_link(0, 9, 0)
