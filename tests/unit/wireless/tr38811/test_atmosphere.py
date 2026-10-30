from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811 import (
    scale_ionospheric_phase_std,
    scale_ionospheric_s4,
    should_compute_gaseous_attenuation,
    slant_gaseous_attenuation_db,
)
from channel_dataset.core import ExecutionContext


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.atmosphere
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_zenith_attenuation_elevation_mapping(backend_name: str) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    elevation = backend.array([pi / 2.0, pi / 6.0], dtype=execution.float_dtype)
    attenuation = slant_gaseous_attenuation_db(0.5, elevation, execution)
    values = np.asarray(attenuation) if backend_name == "numpy" else attenuation.detach().numpy()
    assert np.allclose(values, [0.5, 1.0])


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.atmosphere
def test_gas_applicability_matches_clause_6_6_4() -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    assert should_compute_gaseous_attenuation(20e9, np.array([pi / 2]), execution)
    assert should_compute_gaseous_attenuation(2e9, np.array([5 * pi / 180]), execution)
    assert not should_compute_gaseous_attenuation(2e9, np.array([20 * pi / 180]), execution)
    assert not should_compute_gaseous_attenuation(900e6, np.array([5 * pi / 180]), execution)


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.atmosphere
def test_ionospheric_frequency_scaling_exponents() -> None:
    assert np.isclose(scale_ionospheric_s4(0.2, 1e9, 2e9), 0.2 / (2.0**1.5))
    assert np.isclose(scale_ionospheric_phase_std(0.4, 1e9, 2e9), 0.2)
