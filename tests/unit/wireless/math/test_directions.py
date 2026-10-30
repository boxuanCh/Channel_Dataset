from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless.math import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_spherical_cardinal_directions(execution_context) -> None:
    vectors = spherical_to_cartesian(
        [0.0, pi / 2.0, pi, -pi / 2.0],
        [0.0, 0.0, 0.0, 0.0],
        execution_context,
    )
    assert np.allclose(
        _as_numpy(vectors),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        atol=1.0e-12,
    )


def test_spherical_cartesian_round_trip(execution_context) -> None:
    azimuth = [-2.4, -0.2, 1.7]
    elevation = [-0.6, 0.0, 0.8]
    radius = [1.0, 2.0, 3.0]
    vectors = spherical_to_cartesian(
        azimuth, elevation, execution_context, radius=radius
    )
    recovered_azimuth, recovered_elevation, recovered_radius = cartesian_to_spherical(vectors)
    assert np.allclose(_as_numpy(recovered_azimuth), azimuth, atol=1.0e-12)
    assert np.allclose(_as_numpy(recovered_elevation), elevation, atol=1.0e-12)
    assert np.allclose(_as_numpy(recovered_radius), radius, atol=1.0e-12)


def test_spherical_inputs_broadcast(execution_context) -> None:
    vectors = spherical_to_cartesian(
        [0.0, pi / 2.0], 0.0, execution_context, radius=2.0
    )
    assert np.allclose(_as_numpy(vectors), [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1e-12)


def test_cartesian_zero_vector_is_rejected(backend) -> None:
    with pytest.raises(ValueError, match="zero-length"):
        cartesian_to_spherical(backend.zeros((1, 3), dtype="float64"))


def test_cartesian_final_axis_must_be_xyz(backend) -> None:
    with pytest.raises(ValueError, match="dimension 3"):
        cartesian_to_spherical(backend.ones((2, 2), dtype="float64"))
