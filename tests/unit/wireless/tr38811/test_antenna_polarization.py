from __future__ import annotations

from math import asin, pi

import numpy as np
import pytest

from channel_dataset.channels.wireless.tr38811 import (
    NTNAntennaConfig,
    circular_aperture_power_gain,
    faraday_rotation_matrix,
)
from channel_dataset.core import ExecutionContext


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_circular_aperture_boresight_and_first_null(backend_name: str) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    config = NTNAntennaConfig(1.0, 35.0)
    first_null = asin(3.831705970 / (2.0 * pi))
    gain = circular_aperture_power_gain([0.0, first_null], config, execution)
    values = np.asarray(gain) if backend_name == "numpy" else gain.detach().numpy()
    assert np.isclose(values[0], 1.0)
    assert values[1] < 1e-12


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_faraday_rotation_preserves_vector_power(backend_name: str) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    rotation = faraday_rotation_matrix(0.37, execution)
    values = np.asarray(rotation) if backend_name == "numpy" else rotation.detach().numpy()
    assert np.allclose(values.T @ values, np.eye(2), atol=1e-12)


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.torch
def test_circular_aperture_preserves_torch_gradient() -> None:
    torch = pytest.importorskip("torch")
    execution = ExecutionContext(backend="torch", precision="double", device="cpu")
    angle = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
    gain = circular_aperture_power_gain(angle, NTNAntennaConfig(2.0, 40.0), execution)
    gain.sum().backward()
    assert angle.grad is not None
    assert torch.isfinite(angle.grad).all()
