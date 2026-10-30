from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.ieee80211.config import (
    TGnConfig,
    parse_tgn_config,
)
from channel_dataset.channels.wireless.ieee80211.profiles import (
    load_ieee80211_profile,
    load_tgax_outdoor_profile,
)


def _tgn_parameters() -> dict[str, object]:
    return {
        "profile": "B", "standard_version": "03/940r4",
        "carrier_frequency_hz": 5.25e9, "bandwidth_hz": 20e6,
        "num_rx": 1, "num_rx_antennas": 1, "num_tx": 1,
        "num_tx_antennas": 1, "environmental_speed_mps": 0.0,
        "num_sinusoids": 32, "spatial_quadrature_order": 128,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5, "line_of_sight": False,
        "large_scale_mode": "none", "fluorescent_effect": False,
        "moving_vehicle_effect": False, "normalize_profile_power": True,
    }


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.parametrize("name,count,clusters", [
    ("A", 1, 1), ("B", 9, 2), ("C", 14, 2), ("D", 18, 3),
    ("E", 18, 4), ("F", 18, 6),
])
def test_tgn_resources_are_strict_and_normalized(name: str, count: int, clusters: int) -> None:
    profile = load_ieee80211_profile(name)
    assert len(profile.taps) == count
    assert len(profile.clusters) == clusters
    assert sum(tap.power_linear for tap in profile.taps) == pytest.approx(1.0)
    assert profile.review_status == "pending_independent_review"


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_profile_b_exact_source_values() -> None:
    profile = load_ieee80211_profile("B")
    assert profile.delays_s == pytest.approx(tuple(index * 10e-9 for index in range(9)))
    assert profile.clusters[0].powers_db == (0.0, -5.4, -10.8, -16.2, -21.7)
    assert profile.breakpoint_distance_m == 5.0


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_configuration_rejects_unknown_fields() -> None:
    parameters = _tgn_parameters()
    parameters["typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        parse_tgn_config(parameters)


@pytest.mark.ieee80211
@pytest.mark.tgn
def test_tgn_configuration_validates_applicability() -> None:
    parameters = _tgn_parameters()
    parameters["bandwidth_hz"] = 80e6
    with pytest.raises(ValueError, match="40 MHz"):
        TGnConfig(**parameters)


@pytest.mark.ieee80211
@pytest.mark.tgax
def test_nested_resource_mappings_are_immutable() -> None:
    profile = load_tgax_outdoor_profile()
    with pytest.raises(TypeError):
        profile.scenarios["outdoor_umi"]["los_before"]["distance_coefficient"] = 0.0
