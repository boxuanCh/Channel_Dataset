from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.math import (
    amplitude_db_to_linear,
    linear_amplitude_to_db,
    linear_power_to_db,
    power_db_to_linear,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_power_db_linear_analytic_values_and_round_trip(backend) -> None:
    values_db = backend.array([-20.0, 0.0, 10.0, 20.0], dtype="float64")
    linear = power_db_to_linear(values_db)
    assert np.allclose(_as_numpy(linear), [0.01, 1.0, 10.0, 100.0])
    assert np.allclose(_as_numpy(linear_power_to_db(linear)), _as_numpy(values_db))


def test_amplitude_db_linear_analytic_values_and_round_trip(backend) -> None:
    values_db = backend.array([-20.0, 0.0, 20.0], dtype="float64")
    linear = amplitude_db_to_linear(values_db)
    assert np.allclose(_as_numpy(linear), [0.1, 1.0, 10.0])
    assert np.allclose(_as_numpy(linear_amplitude_to_db(linear)), _as_numpy(values_db))


def test_complex_amplitude_uses_magnitude(backend) -> None:
    values = backend.array([1.0j, -2.0j], dtype="complex128")
    assert np.allclose(_as_numpy(linear_amplitude_to_db(values)), [0.0, 20.0 * np.log10(2.0)])


def test_scalar_conversions() -> None:
    assert power_db_to_linear(30.0) == pytest.approx(1000.0)
    assert linear_power_to_db(1000.0) == pytest.approx(30.0)
    assert amplitude_db_to_linear(6.020599913279624) == pytest.approx(2.0)
    assert linear_amplitude_to_db(-2.0) == pytest.approx(6.020599913279624)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_linear_power_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        linear_power_to_db(value)


def test_zero_amplitude_is_rejected(backend) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        linear_amplitude_to_db(backend.array([1.0, 0.0], dtype="float64"))
