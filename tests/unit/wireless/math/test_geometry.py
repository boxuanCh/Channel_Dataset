from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.math import (
    SPEED_OF_LIGHT_MPS,
    closing_speed_mps,
    doppler_shift_hz,
    euclidean_distance,
    line_of_sight_unit_vector,
    wavelength_m,
)
from channel_dataset.core import Vector3


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_wavelength_and_doppler_analytic_values(backend) -> None:
    frequency = backend.array([1.0e9, 2.0e9], dtype="float64")
    wavelengths = wavelength_m(frequency)
    assert np.allclose(_as_numpy(wavelengths), SPEED_OF_LIGHT_MPS / np.array([1e9, 2e9]))
    shift = doppler_shift_hz(frequency, 10.0)
    assert np.allclose(_as_numpy(shift), 10.0 * np.array([1e9, 2e9]) / SPEED_OF_LIGHT_MPS)


def test_wavelength_rejects_nonpositive_frequency() -> None:
    with pytest.raises(ValueError, match="positive"):
        wavelength_m(0.0)


def test_euclidean_distance_and_unit_vector(execution_context) -> None:
    first = Vector3(0.0, 0.0, 0.0)
    second = Vector3(3.0, 4.0, 0.0)
    distance = euclidean_distance(first, second, execution_context)
    direction = line_of_sight_unit_vector(first, second, execution_context)
    assert float(_as_numpy(distance)) == pytest.approx(5.0)
    assert np.allclose(_as_numpy(direction), [0.6, 0.8, 0.0])


def test_geometry_supports_batches(execution_context) -> None:
    first = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    second = [[0.0, 0.0, 2.0], [4.0, 5.0, 1.0]]
    assert np.allclose(_as_numpy(euclidean_distance(first, second, execution_context)), [2.0, 5.0])


def test_closing_speed_sign_and_tangential_motion(execution_context) -> None:
    transmitter_position = [0.0, 0.0, 0.0]
    receiver_position = [10.0, 0.0, 0.0]
    closing = closing_speed_mps(
        transmitter_position,
        receiver_position,
        [1.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        execution_context,
    )
    tangential = closing_speed_mps(
        transmitter_position,
        receiver_position,
        [0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0],
        execution_context,
    )
    assert float(_as_numpy(closing)) == pytest.approx(3.0)
    assert float(_as_numpy(tangential)) == pytest.approx(0.0)


def test_collocated_line_of_sight_is_rejected(execution_context) -> None:
    with pytest.raises(ValueError, match="collocated"):
        line_of_sight_unit_vector([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], execution_context)
