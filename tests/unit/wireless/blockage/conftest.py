from math import pi

import pytest

from channel_dataset.channels.wireless import ClusterBatch, RayBatch, TopologyBatch


@pytest.fixture
def blockage_case(backend, realization_metadata):
    cluster_shape = (1, 1, 1, 1)
    ray_shape = (1, 1, 1, 1, 3)
    zero_cluster = backend.zeros(cluster_shape, dtype="float64")
    clusters = ClusterBatch(
        zero_cluster,
        backend.ones(cluster_shape, dtype="float64"),
        zero_cluster,
        zero_cluster,
        zero_cluster,
        zero_cluster,
        realization_metadata,
        backend.ones(cluster_shape, dtype="bool"),
    )
    azimuth = backend.array(
        [[[[[-100.0 * pi / 180.0, -60.0 * pi / 180.0, -40.0 * pi / 180.0]]]]],
        dtype="float64",
    )
    zenith = backend.full(ray_shape, 100.0 * pi / 180.0, dtype="float64")
    rays = RayBatch(
        clusters,
        azimuth,
        zenith,
        azimuth,
        zenith,
        backend.ones(ray_shape, dtype="float64"),
        backend.zeros(ray_shape, dtype="float64"),
        ray_mask=backend.ones(ray_shape, dtype="bool"),
        strongest_cluster_indices=backend.zeros((1, 1, 1, 1), dtype="int64"),
    )
    zero = backend.zeros((1, 1, 3), dtype="float64")
    topology = TopologyBatch(
        backend.array([[[100.0, 0.0, 0.0]]], dtype="float64"),
        zero,
        zero,
        zero,
        zero,
        zero,
        backend.zeros((1, 1), dtype="bool"),
        backend.zeros((1, 1), dtype="int64"),
        realization_metadata,
    )
    return rays, topology
