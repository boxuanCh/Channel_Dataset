"""Deterministic helpers for NTN platform state vectors."""

from __future__ import annotations

from math import cos, sin, sqrt
from typing import Any

from channel_dataset.channels.wireless.tr38811.geometry import EARTH_RADIUS_M
from channel_dataset.core import ExecutionContext

EARTH_GRAVITATIONAL_PARAMETER_M3_S2 = 3.986004418e14


def constant_velocity_trajectory(
    initial_position_ecef_m: Any,
    velocity_ecef_mps: Any,
    num_time_samples: int,
    sampling_frequency_hz: float,
    execution: ExecutionContext,
) -> tuple[Any, Any]:
    """Propagate endpoint state vectors linearly over a fixed window."""

    if num_time_samples <= 0 or sampling_frequency_hz <= 0.0:
        raise ValueError("time dimensions and sampling frequency must be positive")
    backend = execution.array_backend
    position = backend.array(
        initial_position_ecef_m, dtype=execution.float_dtype, device=execution.device
    )
    velocity = backend.array(
        velocity_ecef_mps, dtype=execution.float_dtype, device=execution.device
    )
    if backend.shape(position) != backend.shape(velocity) or backend.shape(position)[-1] != 3:
        raise ValueError("initial position and velocity must have equal shapes ending in xyz")
    time = backend.arange(
        num_time_samples, dtype=execution.float_dtype, device=execution.device
    ) / sampling_frequency_hz
    positions = position[..., None, :] + velocity[..., None, :] * time[..., None]
    velocities = backend.broadcast_to(velocity[..., None, :], backend.shape(positions))
    return positions, velocities


def circular_equatorial_orbit(
    altitude_m: float,
    initial_longitude_rad: float,
    num_time_samples: int,
    sampling_frequency_hz: float,
    execution: ExecutionContext,
) -> tuple[Any, Any]:
    """Return a prograde circular equatorial ECEF-like trajectory.

    This deterministic helper omits Earth rotation and is intended for channel
    test windows. Production orbit propagators should supply authoritative ECEF
    state vectors directly.
    """

    if altitude_m <= 0.0:
        raise ValueError("altitude_m must be positive")
    radius = EARTH_RADIUS_M + altitude_m
    angular_rate = sqrt(EARTH_GRAVITATIONAL_PARAMETER_M3_S2 / radius**3)
    backend = execution.array_backend
    time = backend.arange(
        num_time_samples, dtype=execution.float_dtype, device=execution.device
    ) / sampling_frequency_hz
    angle = initial_longitude_rad + angular_rate * time
    zero = backend.zeros((num_time_samples,), dtype=execution.float_dtype, device=execution.device)
    position = backend.stack(
        (radius * backend.cos(angle), radius * backend.sin(angle), zero), axis=-1
    )
    velocity = backend.stack(
        (
            -radius * angular_rate * backend.sin(angle),
            radius * angular_rate * backend.cos(angle),
            zero,
        ),
        axis=-1,
    )
    return position, velocity


def geostationary_state(longitude_rad: float, execution: ExecutionContext) -> tuple[Any, Any]:
    """Return one GEO snapshot at 35,786 km altitude in an ECEF frame."""

    radius = EARTH_RADIUS_M + 35_786_000.0
    backend = execution.array_backend
    position = backend.array(
        [radius * cos(longitude_rad), radius * sin(longitude_rad), 0.0],
        dtype=execution.float_dtype,
        device=execution.device,
    )
    velocity = backend.zeros((3,), dtype=execution.float_dtype, device=execution.device)
    return position, velocity


__all__ = [
    "EARTH_GRAVITATIONAL_PARAMETER_M3_S2", "circular_equatorial_orbit",
    "constant_velocity_trajectory", "geostationary_state",
]
