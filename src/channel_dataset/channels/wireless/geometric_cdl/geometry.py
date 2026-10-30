"""Scatterer construction and backend-native geometric drifting."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.math.constants import SPEED_OF_LIGHT_MPS

LOS_COMPONENT = 0
GROUND_COMPONENT = 1
NLOS_COMPONENT = 2


def _norm(value: Any, backend: Any) -> Any:
    return backend.sqrt(backend.sum(value * value, axis=-1) + 1.0e-30)


def directions_from_angles(azimuth: Any, zenith: Any, backend: Any) -> Any:
    """Return unit vectors for broadcast 3GPP azimuth/zenith angles."""

    sine = backend.sin(zenith)
    return backend.stack(
        (sine * backend.cos(azimuth), sine * backend.sin(azimuth), backend.cos(zenith)),
        axis=-1,
    )


def angles_from_directions(value: Any, backend: Any) -> tuple[Any, Any]:
    """Return wrapped azimuth and zenith from nonzero Cartesian vectors."""

    x, y, z = value[..., 0], value[..., 1], value[..., 2]
    horizontal = backend.sqrt(x * x + y * y + 1.0e-30)
    azimuth = backend.atan2(y, x)
    zenith = backend.atan2(horizontal, z)
    return azimuth, zenith


def _single_bounce(tx: Any, rx: Any, target_length: Any, backend: Any) -> Any:
    """Construct a deterministic ellipsoid point with the target path length."""

    direct = rx - tx
    distance = _norm(direct, backend)
    midpoint = 0.5 * (tx + rx)
    z_axis = direct * 0.0
    z_axis = backend.stack((z_axis[..., 0], z_axis[..., 0], z_axis[..., 0] + 1.0), axis=-1)
    perpendicular = backend.stack(
        (direct[..., 1], -direct[..., 0], direct[..., 0] * 0.0), axis=-1
    )
    perpendicular_norm = _norm(perpendicular, backend)
    alternate = backend.stack(
        (direct[..., 2], direct[..., 2] * 0.0, -direct[..., 0]), axis=-1
    )
    perpendicular = backend.where(
        (perpendicular_norm > 1.0e-9)[..., None], perpendicular, alternate
    )
    perpendicular = perpendicular / (_norm(perpendicular, backend)[..., None] + 1.0e-30)
    half = 0.5 * target_length
    height_squared = half * half - 0.25 * distance * distance
    height = backend.sqrt(
        backend.where(
            height_squared > 0.0,
            height_squared,
            height_squared * 0.0,
        )
        + 1.0e-30
    )
    return midpoint + height[..., None] * perpendicular


def construct_scatterers(
    tx_positions_m: Any,
    rx_positions_m: Any,
    azimuth_departure_rad: Any,
    zenith_departure_rad: Any,
    azimuth_arrival_rad: Any,
    zenith_arrival_rad: Any,
    excess_delay_s: Any,
    component_type_codes: Any,
    *,
    tolerance_m: float,
    max_iterations: int,
    fallback: str,
    minimum_bounce_separation_m: float = 0.0,
) -> tuple[Any, Any, Any, Any]:
    """Solve approximate two-bounce geometry and deterministic fallbacks."""

    backend = get_backend(
        "torch" if type(tx_positions_m).__module__.startswith("torch") else "numpy"
    )
    departure = directions_from_angles(azimuth_departure_rad, zenith_departure_rad, backend)
    arrival = directions_from_angles(azimuth_arrival_rad, zenith_arrival_rad, backend)
    direct = _norm(rx_positions_m - tx_positions_m, backend)
    target = direct + excess_delay_s * SPEED_OF_LIGHT_MPS
    # Start from the closest points on the two directional lines.
    w0 = tx_positions_m - rx_positions_m
    dot_uv = backend.sum(departure * arrival, axis=-1)
    dot_uw = backend.sum(departure * w0, axis=-1)
    dot_vw = backend.sum(arrival * w0, axis=-1)
    denominator = 1.0 - dot_uv * dot_uv
    safe = backend.where(backend.abs(denominator) > 1.0e-8, denominator, denominator * 0.0 + 1.0)
    a = (dot_uv * dot_vw - dot_uw) / safe
    b = (dot_vw - dot_uv * dot_uw) / safe
    a = backend.where(a > 0.0, a, a * 0.0)
    b = backend.where(b > 0.0, b, b * 0.0)
    for _ in range(max_iterations):
        fbs = tx_positions_m + a[..., None] * departure
        lbs = rx_positions_m + b[..., None] * arrival
        length = a + b + _norm(lbs - fbs, backend)
        residual = target - length
        a = backend.where(a + 0.25 * residual > 0.0, a + 0.25 * residual, a * 0.0)
        b = backend.where(b + 0.25 * residual > 0.0, b + 0.25 * residual, b * 0.0)
    fbs = tx_positions_m + a[..., None] * departure
    lbs = rx_positions_m + b[..., None] * arrival
    reconstructed = _norm(fbs - tx_positions_m, backend) + _norm(lbs - fbs, backend) + _norm(rx_positions_m - lbs, backend)
    residual = backend.abs(target - reconstructed)
    bounce_separation = _norm(lbs - fbs, backend)
    valid = (residual <= tolerance_m) & (
        bounce_separation >= minimum_bounce_separation_m
    )
    point = _single_bounce(tx_positions_m, rx_positions_m, target, backend)
    los = component_type_codes == LOS_COMPONENT
    ground = component_type_codes == GROUND_COMPONENT
    fallback_mask = (~valid) & (~los) & (~ground)
    if fallback == "strict" and backend.to_scalar(backend.max(backend.astype(fallback_mask, "int64"))) > 0:
        raise ValueError("two-bounce geometry failed to meet the configured tolerance")
    if fallback in {"single_bounce", "mask"}:
        fbs = backend.where(fallback_mask[..., None], point, fbs)
        lbs = backend.where(fallback_mask[..., None], point, lbs)
    # Specular reflection on the z=0 plane is evaluated with the receiver
    # image. The reflection point remains fixed while the segment drifts.
    rx_image = backend.stack(
        (
            rx_positions_m[..., 0],
            rx_positions_m[..., 1],
            -rx_positions_m[..., 2],
        ),
        axis=-1,
    )
    denominator_z = tx_positions_m[..., 2] + rx_positions_m[..., 2]
    safe_z = backend.where(
        backend.abs(denominator_z) > 1.0e-12,
        denominator_z,
        denominator_z * 0.0 + 1.0,
    )
    reflection_fraction = tx_positions_m[..., 2] / safe_z
    reflection = tx_positions_m + reflection_fraction[..., None] * (
        rx_image - tx_positions_m
    )
    fbs = backend.where(ground[..., None], reflection, fbs)
    lbs = backend.where(ground[..., None], reflection, lbs)
    # Direct components are represented explicitly by the drift kernel.
    fbs = backend.where(los[..., None], rx_positions_m, fbs)
    lbs = backend.where(los[..., None], tx_positions_m, lbs)
    reconstructed = (
        _norm(fbs - tx_positions_m, backend)
        + _norm(lbs - fbs, backend)
        + _norm(rx_positions_m - lbs, backend)
    )
    residual = backend.abs(target - reconstructed)
    return fbs, lbs, residual, fallback_mask


def drift_paths(
    tx_positions_m: Any,
    rx_positions_m: Any,
    fbs_positions_m: Any,
    lbs_positions_m: Any,
    component_type_codes: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Evaluate drifting path length and four global ray angles."""

    backend = get_backend(
        "torch" if type(tx_positions_m).__module__.startswith("torch") else "numpy"
    )
    # Endpoint trajectory axes are [b, endpoint, time, xyz].
    tx = tx_positions_m[:, None, :, None, None, :, :]
    rx = rx_positions_m[:, :, None, None, None, :, :]
    fbs = fbs_positions_m[..., None, :]
    lbs = lbs_positions_m[..., None, :]
    departure_vector = fbs - tx
    arrival_vector = lbs - rx
    path_length = (
        _norm(departure_vector, backend)
        + _norm(lbs - fbs, backend)
        + _norm(rx - lbs, backend)
    )
    direct_vector = rx - tx
    direct_length = _norm(direct_vector, backend)
    los = component_type_codes[..., None, None] == LOS_COMPONENT
    path_length = backend.where(los, direct_length, path_length)
    departure_vector = backend.where(los[..., None], direct_vector, departure_vector)
    # Arrival direction points from the receiver toward the last interaction.
    arrival_vector = backend.where(los[..., None], -direct_vector, arrival_vector)
    aod, zod = angles_from_directions(departure_vector, backend)
    aoa, zoa = angles_from_directions(arrival_vector, backend)
    return path_length, aoa, zoa, aod, zod


__all__ = [
    "GROUND_COMPONENT", "LOS_COMPONENT", "NLOS_COMPONENT", "angles_from_directions",
    "construct_scatterers", "directions_from_angles", "drift_paths",
]
