from __future__ import annotations

import math

import pytest

from channel_dataset.channels.wireless.tr37885 import (
    LOS,
    NLOS,
    NLOSV,
    TR37885TrafficConfig,
    TR37885UrbanGridSimulator,
    V2XSceneSamplingRequest,
    classify_urban_links,
    urban_los_probability,
    urban_path_loss_db,
    vehicle_blockage_parameters_db,
)
from channel_dataset.core import ExecutionContext

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]


def test_los_probability_reference_equation():
    assert urban_los_probability(0.0) == 1.0
    assert urban_los_probability(100.0) == pytest.approx(1.05 * math.exp(-1.14))
    with pytest.raises(ValueError):
        urban_los_probability(-1.0)


def test_path_loss_reference_equations():
    expected_los = 38.77 + 16.7 * math.log10(100.0) + 18.2 * math.log10(6.0)
    expected_nlos = 36.85 + 30.0 * math.log10(100.0) + 18.9 * math.log10(6.0)
    assert urban_path_loss_db(100.0, 6.0e9, LOS) == pytest.approx(expected_los)
    assert urban_path_loss_db(100.0, 6.0e9, NLOSV) == pytest.approx(expected_los)
    assert urban_path_loss_db(100.0, 6.0e9, NLOS) == pytest.approx(expected_nlos)


def test_vehicle_blockage_height_cases():
    assert vehicle_blockage_parameters_db(100.0, 3.1, 3.2, 3.0) == (0.0, 0.0)
    mean, stddev = vehicle_blockage_parameters_db(100.0, 0.75, 1.6, 3.0)
    assert mean == pytest.approx(9.0)
    assert stddev == 4.5
    mean, stddev = vehicle_blockage_parameters_db(100.0, 0.75, 3.0, 1.6)
    assert mean == pytest.approx(5.0)
    assert stddev == 4.0


def test_link_classification_is_seeded_and_persistent():
    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(traffic=TR37885TrafficConfig(num_vehicles=8))
    scene = simulator.generate(V2XSceneSamplingRequest(1, 3, execution, execution.random_stream(4)))
    result = classify_urban_links(
        scene, simulator.grid, (0,), (1,), execution.random_stream(5)
    )
    assert result.condition_codes.shape == (1, 1, 1, 3)
    codes = result.condition_codes[0, 0, 0]
    same_street_codes = [
        int(codes[index]) for index in range(3)
        if bool(result.same_street[0, 0, 0, index])
    ]
    if same_street_codes:
        assert len(set(same_street_codes)) == 1
