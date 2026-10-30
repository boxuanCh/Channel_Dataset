from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.tr37885 import (
    TR37885TrafficConfig,
    TR37885UrbanGrid,
    TR37885UrbanGridConfig,
    TR37885UrbanSystemConfig,
)

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885, pytest.mark.urban_grid]


def test_standard_grid_dimensions_and_counts():
    grid = TR37885UrbanGrid()
    assert grid.config.width_m == 1299.0
    assert grid.config.height_m == 750.0
    assert grid.config.street_width_m == 20.0
    assert len(grid.streets) == 8
    assert len(grid.lanes) == 32
    assert len(grid.intersections) == 16
    assert len(grid.building_blocks) == 9


def test_standard_mode_rejects_physical_override():
    with pytest.raises(ValueError, match="standard mode fixes"):
        TR37885UrbanGridConfig(lane_width_m=4.0)
    with pytest.raises(ValueError, match="standard mode fixes"):
        TR37885TrafficConfig(speed_multiplier=0.9)


def test_controlled_variation_accepts_documented_overrides():
    grid = TR37885UrbanGridConfig(
        mode="controlled_variation", cells_x=4, boundary_policy="periodic"
    )
    traffic = TR37885TrafficConfig(
        mode="controlled_variation", speed_multiplier=0.8,
        vehicle_type_probabilities=(0.3, 0.5, 0.2),
    )
    assert grid.width_m == 1732.0
    assert traffic.resolved_vehicle_type_probabilities == (0.3, 0.5, 0.2)


def test_channel_config_is_strictly_typed():
    assert TR37885UrbanSystemConfig().carrier_frequency_hz == 6.0e9
    with pytest.raises(ValueError, match="standard_version"):
        TR37885UrbanSystemConfig(standard_version="16.0.0")
