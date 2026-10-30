from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811 import free_space_path_loss_db, los_probability
from channel_dataset.core import ExecutionContext


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_los_probability_uses_nearest_elevation_with_upper_midpoint_tie(
    backend_name: str,
) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    elevation = backend.array(
        [10.0, 14.999, 15.0, 90.0], dtype=execution.float_dtype, device=execution.device
    ) * (pi / 180.0)
    result = los_probability(elevation, "dense_urban", execution)
    values = np.asarray(result) if backend_name == "numpy" else result.detach().numpy()
    assert np.allclose(values, [0.282, 0.282, 0.331, 0.981])


@pytest.mark.ntn
@pytest.mark.tr38811
def test_free_space_path_loss_matches_closed_form() -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    distance = np.array([600_000.0], dtype=np.float64)
    result = free_space_path_loss_db(distance, 2.0e9, execution)
    expected = 20.0 * np.log10(distance) + 20.0 * np.log10(2.0e9) - 147.55
    assert np.allclose(result, expected)
