from __future__ import annotations

from math import exp

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    TR37885TrafficConfig,
    TR37885UrbanGridSimulator,
    V2XSceneSamplingRequest,
)
from channel_dataset.core import ExecutionContext

pytestmark = [
    pytest.mark.v2x,
    pytest.mark.tr37885,
    pytest.mark.mobility,
    pytest.mark.statistical,
]


def test_option_b_vehicle_type_mixture_matches_standard_probabilities() -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="B", num_vehicles=300)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(100, 1, execution, execution.random_stream(301))
    )
    values = np.asarray(scene.vehicle_type_codes)
    observed = np.array([(values == code).mean() for code in (1, 2, 3)])
    expected = np.array((0.2, 0.6, 0.2))
    standard_errors = np.sqrt(expected * (1.0 - expected) / values.size)
    assert np.all(np.abs(observed - expected) < 5.0 * standard_errors)


def test_option_a_exponential_bumper_gap_statistics() -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="A", num_vehicles=64)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(500, 1, execution, execution.random_stream(302))
    )
    first = np.asarray(scene.positions_m)[:, 0, 0, :2]
    second = np.asarray(scene.positions_m)[:, 32, 0, :2]
    gaps = np.linalg.norm(second - first, axis=-1) - 5.0
    mean_exponential = (60.0 / 3.6) * 2.0
    expected_mean = 2.0 + mean_exponential * exp(-2.0 / mean_exponential)
    assert np.min(gaps) >= 2.0 - 1.0e-12
    assert np.mean(gaps) == pytest.approx(expected_mean, abs=2.5)
    assert np.std(gaps) > 20.0
