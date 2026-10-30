from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless import ClusterBatch, RayBatch, select_paths


def _array(backend, values, dtype="float64"):
    return backend.array(np.asarray(values), dtype=dtype)


def test_select_paths_converts_one_link_to_semantic_clusters(
    backend, realization_metadata
) -> None:
    cluster_shape = (1, 1, 1, 2)
    clusters = ClusterBatch(
        delays_s=_array(backend, [[[[0.1, 0.2]]]]),
        powers_linear=_array(backend, [[[[0.8, 0.2]]]]),
        azimuth_arrival_rad=backend.zeros(cluster_shape, dtype="float64"),
        zenith_arrival_rad=backend.zeros(cluster_shape, dtype="float64"),
        azimuth_departure_rad=backend.zeros(cluster_shape, dtype="float64"),
        zenith_departure_rad=backend.zeros(cluster_shape, dtype="float64"),
        metadata=realization_metadata,
        cluster_mask=_array(backend, [[[[True, True]]]], "bool"),
    )
    ray_shape = (1, 1, 1, 2, 2)
    zeros = backend.zeros(ray_shape, dtype="float64")
    rays = RayBatch(
        clusters=clusters,
        azimuth_arrival_rad=_array(backend, [[[[[0.1, 0.2], [0.3, 0.4]]]]]),
        zenith_arrival_rad=backend.full(ray_shape, pi / 2.0, dtype="float64"),
        azimuth_departure_rad=zeros,
        zenith_departure_rad=backend.full(ray_shape, pi / 2.0, dtype="float64"),
        xpr_linear=backend.full(ray_shape, 10.0, dtype="float64"),
        initial_phase_rad=_array(backend, [[[[[0.0, pi / 2.0], [pi, 0.0]]]]]),
        ray_mask=_array(backend, [[[[[True, True], [True, False]]]]], "bool"),
        blockage_loss_db=_array(backend, [[[[[0.0, 0.0], [20.0, 0.0]]]]]),
        strongest_cluster_indices=backend.zeros((1, 1, 1, 1), dtype="int64"),
    )

    converted = select_paths(rays, 0, 0, 0)
    assert len(converted) == 2
    assert len(converted[0].paths) == 2
    assert len(converted[1].paths) == 1
    assert converted[0].metadata["strongest"] is True
    assert converted[0].paths[0].delay_s == pytest.approx(0.1)
    assert converted[0].paths[0].arrival is not None
    assert converted[0].paths[0].arrival.elevation_rad == pytest.approx(0.0)
    assert abs(converted[0].paths[0].complex_gain) == pytest.approx(np.sqrt(0.4))
    assert abs(converted[1].paths[0].complex_gain) == pytest.approx(np.sqrt(0.2) * 0.1)


def test_select_paths_rejects_valid_cluster_without_rays(backend, realization_metadata) -> None:
    shape = (1, 1, 1, 1)
    ray_shape = (*shape, 1)
    zeros = backend.zeros(shape, dtype="float64")
    clusters = ClusterBatch(
        zeros,
        backend.ones(shape, dtype="float64"),
        zeros,
        zeros,
        zeros,
        zeros,
        realization_metadata,
        backend.ones(shape, dtype="bool"),
    )
    ray_zeros = backend.zeros(ray_shape, dtype="float64")
    rays = RayBatch(
        clusters,
        ray_zeros,
        ray_zeros,
        ray_zeros,
        ray_zeros,
        backend.ones(ray_shape, dtype="float64"),
        ray_zeros,
        ray_mask=backend.zeros(ray_shape, dtype="bool"),
    )
    with pytest.raises(ValueError, match="valid cluster"):
        select_paths(rays, 0, 0, 0)
