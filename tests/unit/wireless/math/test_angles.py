from __future__ import annotations

from math import pi

import numpy as np

from channel_dataset.channels.wireless.math import (
    degrees_to_radians,
    radians_to_degrees,
    wrap_angle_degrees,
    wrap_angle_radians,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_degree_radian_round_trip(backend) -> None:
    degrees = backend.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype="float64")
    radians = degrees_to_radians(degrees)
    assert np.allclose(_as_numpy(radians), [-pi, -pi / 2.0, 0.0, pi / 2.0, pi])
    assert np.allclose(_as_numpy(radians_to_degrees(radians)), _as_numpy(degrees))


def test_radian_wrapping_has_half_open_periodic_boundaries(backend) -> None:
    angles = backend.array([-3.0 * pi, -pi, pi, 3.0 * pi, 5.0 * pi / 2.0], dtype="float64")
    wrapped = wrap_angle_radians(angles)
    assert np.allclose(_as_numpy(wrapped), [-pi, -pi, -pi, -pi, pi / 2.0])


def test_degree_wrapping_with_zero_lower_bound(backend) -> None:
    angles = backend.array([-720.0, -90.0, 0.0, 360.0, 450.0], dtype="float64")
    wrapped = wrap_angle_degrees(angles, lower_bound=0.0)
    assert np.allclose(_as_numpy(wrapped), [0.0, 270.0, 0.0, 0.0, 90.0])
