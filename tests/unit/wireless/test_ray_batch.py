import numpy as np
import pytest

from channel_dataset.channels.wireless import ClusterBatch, RayBatch


def _ray_batch(backend, metadata) -> RayBatch:
    cluster_shape = (2, 3, 2, 3)
    ray_shape = (*cluster_shape, 4)
    cluster_zeros = backend.zeros(cluster_shape, dtype="float64")
    clusters = ClusterBatch(
        cluster_zeros,
        backend.full(cluster_shape, 1.0 / 3.0, dtype="float64"),
        cluster_zeros,
        cluster_zeros,
        cluster_zeros,
        cluster_zeros,
        metadata,
        backend.ones(cluster_shape, dtype="bool"),
    )
    ray_zeros = backend.zeros(ray_shape, dtype="float64")
    sort_indices = backend.array(
        np.broadcast_to(np.arange(3), cluster_shape).copy(), dtype="int64"
    )
    strongest = backend.zeros((2, 3, 2, 1), dtype="int64")
    return RayBatch(
        clusters=clusters,
        azimuth_arrival_rad=ray_zeros,
        zenith_arrival_rad=ray_zeros,
        azimuth_departure_rad=ray_zeros,
        zenith_departure_rad=ray_zeros,
        xpr_linear=backend.ones(ray_shape, dtype="float64"),
        initial_phase_rad=ray_zeros,
        ray_mask=backend.ones(ray_shape, dtype="bool"),
        blockage_loss_db=ray_zeros,
        sort_indices=sort_indices,
        strongest_cluster_indices=strongest,
    )


def test_ray_batch_validates_and_selects_link_view(backend, realization_metadata) -> None:
    rays = _ray_batch(backend, realization_metadata)
    selected = rays.select_link(1, 2, 1)
    assert rays.shape == (2, 3, 2, 3, 4)
    assert selected.shape == (1, 1, 1, 3, 4)
    assert selected.clusters.shape == (1, 1, 1, 3)
    if type(rays.initial_phase_rad).__module__.partition(".")[0] == "torch":
        assert (
            rays.initial_phase_rad.untyped_storage().data_ptr()
            == selected.initial_phase_rad.untyped_storage().data_ptr()
        )
    else:
        assert np.shares_memory(rays.initial_phase_rad, selected.initial_phase_rad)


def test_ray_batch_rejects_cluster_mismatch(backend, realization_metadata) -> None:
    rays = _ray_batch(backend, realization_metadata)
    wrong = backend.zeros((2, 3, 2, 2, 4), dtype="float64")
    with pytest.raises(ValueError, match="match clusters"):
        RayBatch(
            rays.clusters,
            wrong,
            wrong,
            wrong,
            wrong,
            backend.ones((2, 3, 2, 2, 4), dtype="float64"),
            wrong,
        )


def test_ray_batch_rejects_invalid_xpr_and_indices(backend, realization_metadata) -> None:
    rays = _ray_batch(backend, realization_metadata)
    with pytest.raises(ValueError, match="xpr_linear"):
        RayBatch(
            rays.clusters,
            rays.azimuth_arrival_rad,
            rays.zenith_arrival_rad,
            rays.azimuth_departure_rad,
            rays.zenith_departure_rad,
            backend.zeros(rays.shape, dtype="float64"),
            rays.initial_phase_rad,
        )
    with pytest.raises(ValueError, match="sort_indices"):
        RayBatch(
            rays.clusters,
            rays.azimuth_arrival_rad,
            rays.zenith_arrival_rad,
            rays.azimuth_departure_rad,
            rays.zenith_departure_rad,
            rays.xpr_linear,
            rays.initial_phase_rad,
            sort_indices=backend.full(rays.clusters.shape, 9, dtype="int64"),
        )
