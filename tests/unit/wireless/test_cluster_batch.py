import numpy as np
import pytest

from channel_dataset.channels.wireless import ClusterBatch


def _clusters(backend, metadata, shape=(2, 3, 2, 4)) -> ClusterBatch:
    zeros = backend.zeros(shape, dtype="float64")
    powers = backend.full(shape, 1.0 / shape[-1], dtype="float64")
    return ClusterBatch(
        delays_s=zeros,
        powers_linear=powers,
        azimuth_arrival_rad=zeros,
        zenith_arrival_rad=zeros,
        azimuth_departure_rad=zeros,
        zenith_departure_rad=zeros,
        metadata=metadata,
        cluster_mask=backend.ones(shape, dtype="bool"),
    )


def test_cluster_batch_selects_link_view(backend, realization_metadata) -> None:
    clusters = _clusters(backend, realization_metadata)
    selected = clusters.select_link(1, 2, 1)
    assert clusters.shape == (2, 3, 2, 4)
    assert selected.shape == (1, 1, 1, 4)
    if type(clusters.delays_s).__module__.partition(".")[0] == "torch":
        assert (
            clusters.delays_s.untyped_storage().data_ptr()
            == selected.delays_s.untyped_storage().data_ptr()
        )
    else:
        assert np.shares_memory(clusters.delays_s, selected.delays_s)


def test_cluster_batch_rejects_shape_mismatch(backend, realization_metadata) -> None:
    clusters = _clusters(backend, realization_metadata, (1, 1, 1, 2))
    with pytest.raises(ValueError, match="powers_linear"):
        ClusterBatch(
            clusters.delays_s,
            backend.ones((1, 1, 1, 3), dtype="float64"),
            clusters.azimuth_arrival_rad,
            clusters.zenith_arrival_rad,
            clusters.azimuth_departure_rad,
            clusters.zenith_departure_rad,
            realization_metadata,
        )


def test_cluster_batch_rejects_negative_power(backend, realization_metadata) -> None:
    shape = (1, 1, 1, 1)
    zeros = backend.zeros(shape, dtype="float64")
    with pytest.raises(ValueError, match="powers_linear"):
        ClusterBatch(
            zeros,
            backend.full(shape, -1.0, dtype="float64"),
            zeros,
            zeros,
            zeros,
            zeros,
            realization_metadata,
        )
