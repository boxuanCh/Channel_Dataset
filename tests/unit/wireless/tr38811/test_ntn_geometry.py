from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811 import (
    EARTH_RADIUS_M,
    compute_ntn_geometry,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from channel_dataset.core import ExecutionContext


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_zenith_geometry_has_expected_range_delay_and_elevation(
    backend_name: str, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    topology = zenith_topology_factory(execution)
    geometry = compute_ntn_geometry(topology, 4, 2.0e9)
    to_numpy = lambda value: (
        np.asarray(value) if backend_name == "numpy" else value.detach().numpy()
    )
    assert np.allclose(to_numpy(geometry.slant_range_m), 600_000.0)
    assert np.allclose(to_numpy(geometry.elevation_rad), pi / 2.0)
    assert np.allclose(
        to_numpy(geometry.propagation_delay_s), 600_000.0 / 299_792_458.0
    )
    assert np.allclose(to_numpy(geometry.common_doppler_hz), 0.0)
    assert np.all(to_numpy(geometry.visibility))


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_geodetic_round_trip_on_spherical_earth(backend_name: str) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    position = geodetic_to_ecef(0.4, -1.2, 1234.0, execution)
    latitude, longitude, altitude = ecef_to_geodetic(position, execution)
    assert np.isclose(backend.to_scalar(latitude), 0.4)
    assert np.isclose(backend.to_scalar(longitude), -1.2)
    assert np.isclose(backend.to_scalar(altitude), 1234.0)
    values = np.asarray(position) if backend_name == "numpy" else position.detach().numpy()
    assert np.isclose(np.linalg.norm(values), EARTH_RADIUS_M + 1234.0)
