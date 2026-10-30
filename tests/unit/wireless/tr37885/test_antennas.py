from __future__ import annotations

from math import pi

import pytest

from channel_dataset.channels.wireless import load_vehicle_antenna_profile

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]


def test_type_one_and_three_use_front_and_rear_option_one_locations() -> None:
    for vehicle_type in (1, 3):
        profile = load_vehicle_antenna_profile(vehicle_type)
        assert profile.locations == ("front", "rear")
        assert profile.bearing_rad == pytest.approx((0.0, pi))
        assert profile.baseline_array == ((1, 1, 2, 1, 1),) * 2
        assert profile.maximum_element_gain_db == 3.0


def test_type_two_uses_rooftop_option_one_array() -> None:
    profile = load_vehicle_antenna_profile(2)
    assert profile.locations == ("rooftop",)
    assert profile.baseline_array == ((1, 2, 2, 1, 1),)
    assert profile.optional_array == ((1, 4, 2, 1, 1),)
    assert profile.element_spacing_wavelengths == 0.5
    assert profile.tilt_rad == pytest.approx(pi / 2.0)


def test_vehicle_yaw_rotates_front_and_rear_bearings() -> None:
    profile = load_vehicle_antenna_profile(1)
    assert profile.port_bearings_rad(4, pi / 2.0) == pytest.approx(
        (pi / 2.0, 3.0 * pi / 2.0, pi / 2.0, 3.0 * pi / 2.0)
    )


def test_invalid_vehicle_antenna_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="1, 2, or 3"):
        load_vehicle_antenna_profile(4)
