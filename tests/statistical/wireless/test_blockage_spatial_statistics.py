from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    ClusterBatch,
    NonSelfBlockingModelA,
    RayBatch,
    SnapshotSpatialConsistency,
    TopologyBatch,
    load_blockage_parameters,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = pytest.mark.statistical


def _metadata() -> RealizationMetadata:
    return RealizationMetadata(
        "wireless", "blockage_statistics", 1, "numpy", "double", "cpu",
        "numpy:79000:root", "0" * 64, {}, "test",
    )


def _one_ray_case(batch: int) -> tuple[RayBatch, TopologyBatch]:
    execution = ExecutionContext("numpy", "double")
    backend = execution.array_backend
    metadata = _metadata()
    cluster_shape = (batch, 1, 1, 1)
    ray_shape = (*cluster_shape, 1)
    zero_cluster = backend.zeros(cluster_shape, dtype="float64")
    clusters = ClusterBatch(
        zero_cluster,
        backend.ones(cluster_shape, dtype="float64"),
        zero_cluster,
        backend.full(cluster_shape, pi / 2.0, dtype="float64"),
        zero_cluster,
        backend.full(cluster_shape, pi / 2.0, dtype="float64"),
        metadata,
        backend.ones(cluster_shape, dtype="bool"),
    )
    rays = RayBatch(
        clusters,
        backend.zeros(ray_shape, dtype="float64"),
        backend.full(ray_shape, pi / 2.0, dtype="float64"),
        backend.zeros(ray_shape, dtype="float64"),
        backend.full(ray_shape, pi / 2.0, dtype="float64"),
        backend.ones(ray_shape, dtype="float64"),
        backend.zeros(ray_shape, dtype="float64"),
        ray_mask=backend.ones(ray_shape, dtype="bool"),
    )
    zeros = backend.zeros((batch, 1, 3), dtype="float64")
    topology = TopologyBatch(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        backend.zeros((batch, 1), dtype="bool"),
        backend.zeros((batch, 1), dtype="int64"),
        metadata,
    )
    return rays, topology


@pytest.mark.parametrize("mode", ["portrait", "landscape"])
def test_model_a_overlap_count_matches_binomial_geometry(mode: str) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 20_000
    rays, topology = _one_ray_case(batch)
    parameters = load_blockage_parameters()["model_a"][mode]
    loss = as_numpy(
        NonSelfBlockingModelA(mode).loss_db(
            rays, topology, execution.random_stream(79_001)
        )
    ).reshape(-1)
    overlap_count = loss / float(parameters["attenuation_db"])
    probability = float(parameters["azimuth_half_width_degrees"]) / 180.0
    blockers = int(parameters["num_blockers"])
    expected_mean = blockers * probability
    expected_std = np.sqrt(blockers * probability * (1.0 - probability))
    assert_mean(overlap_count, expected_mean, expected_std)
    assert_standard_deviation(overlap_count, expected_std, relative_floor=0.015)
    assert np.all(overlap_count == np.floor(overlap_count))


def test_snapshot_spatial_consistency_matches_full_covariance_and_independence() -> None:
    execution = ExecutionContext("numpy", "double")
    backend = execution.array_backend
    batch = 5_000
    distance = 10.0
    positions = np.array(
        [[0.0, 0.0], [0.5 * distance, 0.0], [distance, 0.0], [2.0 * distance, 0.0]]
    )
    positions = backend.array(np.tile(positions[None, ...], (batch, 1, 1)), dtype="float64")
    result = SnapshotSpatialConsistency(distance).sample(
        positions,
        execution.random_stream(79_002),
        num_lsp_fields=7,
        num_small_scale_fields=4,
    )
    lsp = as_numpy(result.lsp_gaussian)
    expected = np.exp(-np.abs(np.subtract.outer((0.0, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0))))
    for field in range(lsp.shape[-1]):
        observed = np.corrcoef(lsp[..., field], rowvar=False)
        assert np.max(np.abs(observed - expected)) < 0.055
        assert abs(float(np.mean(lsp[..., field]))) < 0.035
        assert abs(float(np.std(lsp[..., field])) - 1.0) < 0.035
    colocated_fields = lsp[:, 0, :]
    cross = np.corrcoef(colocated_fields, rowvar=False)
    off_diagonal = cross[~np.eye(cross.shape[0], dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.055
    uniform = np.sort(as_numpy(result.los_uniform)[:, 0])
    expected_order = (np.arange(batch) + 0.5) / batch
    assert np.max(np.abs(uniform - expected_order)) < 0.025
