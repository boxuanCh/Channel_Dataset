from math import pi

from channel_dataset.channels.wireless import circular_angular_spread


def test_angles_are_in_canonical_ranges(backend, ray_case):
    _, _, _, rays = ray_case
    for value in (rays.azimuth_arrival_rad, rays.azimuth_departure_rad):
        assert backend.to_scalar(backend.min(value)) >= -pi
        assert backend.to_scalar(backend.max(value)) <= pi


def test_circular_spread_is_finite_per_link(backend, ray_case):
    _, _, _, rays = ray_case
    spread = circular_angular_spread(rays)
    assert backend.shape(spread) == rays.shape[:3]
    assert backend.all_finite(spread)
    assert backend.all_nonnegative(spread)
    for value in (rays.zenith_arrival_rad, rays.zenith_departure_rad):
        assert backend.to_scalar(backend.min(value)) >= 0.0
        assert backend.to_scalar(backend.max(value)) <= pi
