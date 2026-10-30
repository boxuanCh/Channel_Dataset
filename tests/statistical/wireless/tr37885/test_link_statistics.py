from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    TR37885UrbanGrid,
    V2XSceneTrajectory,
    classify_urban_links,
    urban_los_probability,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885, pytest.mark.statistical]


def test_same_street_los_frequency_matches_standard_probability() -> None:
    batch_size = 2000
    execution = ExecutionContext("numpy", "double", "cpu")
    backend = execution.array_backend
    positions = np.zeros((batch_size, 2, 1, 3), dtype=np.float64)
    positions[:, 0, 0, :] = (50.0, 1.75, 1.6)
    positions[:, 1, 0, :] = (150.0, 1.75, 1.6)
    metadata = RealizationMetadata(
        "wireless", "tr37885_test_scene", 1, "numpy", "double", "cpu",
        "numpy:44:root", "0" * 64,
        {"grid": {"mode": "standard"}}, "test", root_seed=44,
        standard_version="15.3.0",
    )
    scene = V2XSceneTrajectory(
        backend.array(positions, dtype="float64"),
        backend.zeros((batch_size, 2, 1, 3), dtype="float64"),
        backend.zeros((batch_size, 2, 1, 3), dtype="float64"),
        backend.ones((batch_size, 2, 1), dtype="bool"),
        backend.zeros((batch_size, 2, 1), dtype="int64"),
        backend.zeros((batch_size, 2, 1), dtype="int64"),
        backend.array(np.tile((0, 1), (batch_size, 1)), dtype="int64"),
        backend.full((batch_size, 2), 2, dtype="int64"),
        backend.array(np.tile((5.0, 2.0, 1.6), (batch_size, 2, 1)), dtype="float64"),
        backend.full((batch_size, 2), 1.6, dtype="float64"),
        backend.zeros((1,), dtype="float64"),
        metadata,
    )
    conditions = classify_urban_links(
        scene, TR37885UrbanGrid(), (0,), (1,), execution.random_stream(44), metadata=metadata
    )
    observed = np.mean(np.asarray(conditions.condition_codes) == 0)
    expected = urban_los_probability(100.0)
    standard_error = np.sqrt(expected * (1.0 - expected) / batch_size)
    assert observed == pytest.approx(expected, abs=5.0 * standard_error)
