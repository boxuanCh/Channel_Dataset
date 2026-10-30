from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.geometric_cdl import SpatialFieldBank
from channel_dataset.core import ExecutionContext
from tests.statistical.statistical_assertions import assert_mean, assert_standard_deviation

pytestmark = pytest.mark.statistical


def _target(acf: str, distance_m: float, correlation_distance_m: float) -> float:
    scaled = distance_m / correlation_distance_m
    exponential = np.exp(-scaled)
    gaussian = np.exp(-(scaled**2))
    if acf == "exponential":
        return float(exponential)
    if acf == "gaussian":
        return float(gaussian)
    return float(0.5 * (exponential + gaussian))


@pytest.mark.parametrize("acf", ["exponential", "gaussian", "gaussian_exponential"])
def test_sos_field_matches_configured_spatial_acf(acf: str) -> None:
    execution = ExecutionContext("numpy", "double")
    field_count = 4_096
    correlation_distance = 10.0
    bank = SpatialFieldBank.sample(
        field_count,
        300,
        correlation_distance,
        execution.random_stream(76_000),
        dtype="float64",
        acf=acf,
    )
    distances = np.asarray((0.0, 1.0, 5.0, 10.0, 20.0))
    positions = np.column_stack((distances, np.zeros_like(distances), np.zeros_like(distances)))
    samples = np.asarray(bank.evaluate(positions))
    observed = np.corrcoef(samples)[0]
    expected = np.asarray([_target(acf, distance, correlation_distance) for distance in distances])
    # 4096 independent field phases give a Fisher correlation SE near 0.016;
    # 0.06 is a conservative approximation allowance for the finite SOS basis.
    assert np.max(np.abs(observed - expected)) < 0.06


def test_sos_field_normal_and_uniform_marginals() -> None:
    execution = ExecutionContext("numpy", "double")
    bank = SpatialFieldBank.sample(
        10_000, 300, 10.0, execution.random_stream(76_001), dtype="float64"
    )
    normal = np.asarray(bank.evaluate(np.zeros((1, 3))))[0]
    assert_mean(normal, 0.0, 1.0)
    assert_standard_deviation(normal, 1.0)
    uniform = np.sort(np.asarray(bank.normal_to_uniform(normal)))
    expected_order = (np.arange(uniform.size) + 0.5) / uniform.size
    assert abs(float(np.mean(uniform)) - 0.5) < 0.01
    assert abs(float(np.var(uniform)) - 1.0 / 12.0) < 0.005
    assert np.max(np.abs(uniform - expected_order)) < 0.02
