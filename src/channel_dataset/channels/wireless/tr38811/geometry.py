"""Earth-space geometry and motion for TR 38.811 service links."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.tr38811.types import NTNLinkGeometry, NTNTopologyBatch
from channel_dataset.core import ExecutionContext, RealizationMetadata

EARTH_RADIUS_M = 6_371_000.0


def geodetic_to_ecef(
    latitude_rad: Any,
    longitude_rad: Any,
    altitude_m: Any,
    execution: ExecutionContext,
) -> Any:
    """Convert spherical-Earth geodetic coordinates to ECEF metres."""

    backend = execution.array_backend
    latitude = backend.array(latitude_rad, dtype=execution.float_dtype, device=execution.device)
    longitude = backend.array(
        longitude_rad, dtype=execution.float_dtype, device=execution.device
    )
    altitude = backend.array(altitude_m, dtype=execution.float_dtype, device=execution.device)
    radius = EARTH_RADIUS_M + altitude
    cos_latitude = backend.cos(latitude)
    return backend.stack(
        (
            radius * cos_latitude * backend.cos(longitude),
            radius * cos_latitude * backend.sin(longitude),
            radius * backend.sin(latitude),
        ),
        axis=-1,
    )


def ecef_to_geodetic(position_ecef_m: Any, execution: ExecutionContext) -> tuple[Any, Any, Any]:
    """Convert ECEF metres to spherical-Earth latitude, longitude, and altitude."""

    backend = execution.array_backend
    position = backend.array(
        position_ecef_m, dtype=execution.float_dtype, device=execution.device
    )
    if backend.shape(position)[-1] != 3:
        raise ValueError("position_ecef_m must end with an xyz axis")
    x, y, z = position[..., 0], position[..., 1], position[..., 2]
    horizontal = backend.sqrt(x * x + y * y)
    radius = backend.sqrt(horizontal * horizontal + z * z)
    return backend.atan2(z, horizontal), backend.atan2(y, x), radius - EARTH_RADIUS_M


def _expand_trajectory(value: Any, samples: int, backend: Any) -> Any:
    shape = backend.shape(value)
    if shape[2] == samples:
        return value
    if shape[2] != 1:
        raise ValueError("trajectory length must be one or num_time_samples")
    return backend.broadcast_to(value, (shape[0], shape[1], samples, 3))


def compute_ntn_geometry(
    topology: NTNTopologyBatch,
    num_time_samples: int,
    carrier_frequency_hz: float,
    *,
    metadata: RealizationMetadata | None = None,
) -> NTNLinkGeometry:
    """Derive range, angles, delay, radial velocity, and Doppler.

    Positive radial velocity and positive Doppler denote closing endpoints.
    Azimuth and elevation are measured at the ground endpoint toward the
    satellite or HAPS endpoint.
    """

    if num_time_samples <= 0:
        raise ValueError("num_time_samples must be positive")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive")
    backend = get_backend(topology.metadata.backend)
    tx_position = _expand_trajectory(topology.tx_positions_ecef_m, num_time_samples, backend)
    rx_position = _expand_trajectory(topology.rx_positions_ecef_m, num_time_samples, backend)
    tx_velocity = _expand_trajectory(topology.tx_velocities_ecef_mps, num_time_samples, backend)
    rx_velocity = _expand_trajectory(topology.rx_velocities_ecef_mps, num_time_samples, backend)

    displacement = tx_position[:, None, :, :, :] - rx_position[:, :, None, :, :]
    slant_range = backend.sqrt(backend.sum(displacement * displacement, axis=-1))
    direction_rx_to_tx = displacement / slant_range[..., None]
    relative_velocity = tx_velocity[:, None, :, :, :] - rx_velocity[:, :, None, :, :]
    range_rate = backend.sum(relative_velocity * direction_rx_to_tx, axis=-1)
    closing_velocity = -range_rate
    common_doppler = closing_velocity * carrier_frequency_hz / SPEED_OF_LIGHT_MPS

    if topology.rx_endpoint_kind == "ground":
        ground_position = rx_position[:, :, None, :, :]
        platform_position = tx_position[:, None, :, :, :]
        ground_to_platform = direction_rx_to_tx
    else:
        ground_position = tx_position[:, None, :, :, :]
        platform_position = rx_position[:, :, None, :, :]
        ground_to_platform = -direction_rx_to_tx

    ground_radius = backend.sqrt(backend.sum(ground_position * ground_position, axis=-1))
    platform_radius = backend.sqrt(backend.sum(platform_position * platform_position, axis=-1))
    up = ground_position / ground_radius[..., None]
    platform_up = platform_position / platform_radius[..., None]
    vertical = backend.sum(ground_to_platform * up, axis=-1)
    horizontal = backend.sqrt(backend.clip(1.0 - vertical * vertical, 0.0, 1.0))
    elevation = backend.atan2(vertical, horizontal)

    ground_x, ground_y, ground_z = (
        ground_position[..., 0], ground_position[..., 1], ground_position[..., 2]
    )
    longitude = backend.atan2(ground_y, ground_x)
    latitude = backend.atan2(
        ground_z, backend.sqrt(ground_x * ground_x + ground_y * ground_y)
    )
    float_dtype = "float64" if topology.metadata.precision == "double" else "float32"
    zero = backend.zeros(
        backend.shape(longitude), dtype=float_dtype, device=topology.metadata.device
    )
    east = backend.stack((-backend.sin(longitude), backend.cos(longitude), zero), axis=-1)
    north = backend.stack(
        (
            -backend.sin(latitude) * backend.cos(longitude),
            -backend.sin(latitude) * backend.sin(longitude),
            backend.cos(latitude),
        ),
        axis=-1,
    )
    azimuth = backend.atan2(
        backend.sum(ground_to_platform * east, axis=-1),
        backend.sum(ground_to_platform * north, axis=-1),
    )
    azimuth = backend.where(azimuth < 0.0, azimuth + 2.0 * pi, azimuth)

    cosine_central = backend.clip(backend.sum(up * platform_up, axis=-1), -1.0, 1.0)
    sine_central = backend.sqrt(backend.clip(1.0 - cosine_central * cosine_central, 0.0, 1.0))
    central_angle = backend.atan2(sine_central, cosine_central)
    visibility = elevation >= 0.0
    if topology.link_mask is not None:
        mask = topology.link_mask[..., None]
        visibility = visibility & backend.broadcast_to(mask, backend.shape(visibility))

    return NTNLinkGeometry(
        slant_range_m=slant_range,
        elevation_rad=elevation,
        azimuth_rad=azimuth,
        central_angle_rad=central_angle,
        propagation_delay_s=slant_range / SPEED_OF_LIGHT_MPS,
        radial_velocity_mps=closing_velocity,
        common_doppler_hz=common_doppler,
        visibility=visibility,
        direction_rx_to_tx=direction_rx_to_tx,
        metadata=topology.metadata if metadata is None else metadata,
    )


def doppler_phase(common_doppler_hz: Any, sampling_frequency_hz: float, execution: ExecutionContext) -> Any:
    """Integrate a time-varying Doppler trajectory into complex phase."""

    if sampling_frequency_hz <= 0.0:
        raise ValueError("sampling_frequency_hz must be positive")
    backend = execution.array_backend
    samples = backend.shape(common_doppler_hz)[-1]
    zero = backend.zeros(
        (*backend.shape(common_doppler_hz)[:-1], 1),
        dtype=execution.float_dtype,
        device=execution.device,
    )
    if samples == 1:
        phase_cycles = zero
    else:
        increments = 0.5 * (
            common_doppler_hz[..., 1:] + common_doppler_hz[..., :-1]
        ) / sampling_frequency_hz
        phase_cycles = backend.concatenate((zero, backend.cumsum(increments, axis=-1)), axis=-1)
    return backend.exp(1j * 2.0 * pi * phase_cycles)


__all__ = [
    "EARTH_RADIUS_M", "compute_ntn_geometry", "doppler_phase", "ecef_to_geodetic",
    "geodetic_to_ecef",
]
