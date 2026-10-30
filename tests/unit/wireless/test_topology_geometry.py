import numpy as np
import pytest

from channel_dataset.channels.wireless import TopologyBatch, compute_link_geometry


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _topology(backend, metadata, tx, rx):
    tx = backend.array(tx, dtype="float64")
    rx = backend.array(rx, dtype="float64")
    return TopologyBatch(
        tx_positions_m=tx,
        rx_positions_m=rx,
        tx_velocities_mps=backend.zeros(tx.shape, dtype="float64"),
        rx_velocities_mps=backend.zeros(rx.shape, dtype="float64"),
        tx_orientations_rad=backend.zeros(tx.shape, dtype="float64"),
        rx_orientations_rad=backend.zeros(rx.shape, dtype="float64"),
        rx_indoor=backend.zeros(rx.shape[:2], dtype="bool"),
        tx_site_ids=backend.zeros(tx.shape[:2], dtype="int64"),
        metadata=metadata,
    )


def test_hand_calculated_geometry_and_quadrants(backend, realization_metadata) -> None:
    topology = _topology(
        backend,
        realization_metadata,
        [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
        [[[3.0, 4.0, 12.0]]],
    )
    geometry = compute_link_geometry(topology)
    assert np.allclose(_numpy(geometry.distance_2d_m), [[[5.0, np.sqrt(13.0)]]])
    assert np.allclose(_numpy(geometry.distance_3d_m)[0, 0, 0], 13.0)
    assert np.allclose(_numpy(geometry.azimuth_departure_rad)[0, 0, 0], np.arctan2(4, 3))
    assert np.allclose(_numpy(geometry.azimuth_departure_rad)[0, 0, 1], np.arctan2(2, 3))


def test_reciprocal_geometry_transposes_links(backend, realization_metadata) -> None:
    topology = _topology(
        backend,
        realization_metadata,
        [[[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]],
        [[[1.0, 3.0, 2.0]]],
    )
    forward = compute_link_geometry(topology)
    reverse = forward.reverse()
    assert reverse.shape == (1, 2, 1)
    assert np.array_equal(
        _numpy(reverse.azimuth_departure_rad),
        np.transpose(_numpy(forward.azimuth_arrival_rad), (0, 2, 1)),
    )


def test_collocated_geometry_is_rejected(backend, realization_metadata) -> None:
    topology = _topology(
        backend, realization_metadata, [[[1.0, 2.0, 3.0]]], [[[1.0, 2.0, 3.0]]]
    )
    with pytest.raises(ValueError, match="collocated"):
        compute_link_geometry(topology)
