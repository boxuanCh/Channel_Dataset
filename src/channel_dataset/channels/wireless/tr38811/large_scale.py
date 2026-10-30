"""TR 38.811 LOS probability, path loss, clutter, and shadow fading."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.backends import RandomStream
from channel_dataset.channels.wireless.tr38811.config import TR38811DeploymentConfig
from channel_dataset.channels.wireless.tr38811.profiles import load_resource
from channel_dataset.channels.wireless.tr38811.types import (
    NTNLargeScaleComponents,
    NTNLinkGeometry,
    NTNTopologyBatch,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata


def _nearest_table_value(elevation_rad: Any, values: list[float], execution: ExecutionContext) -> Any:
    backend = execution.array_backend
    degrees = elevation_rad * (180.0 / pi)
    result = backend.full(
        backend.shape(degrees), values[-1], dtype=execution.float_dtype, device=execution.device
    )
    for index in range(7, -1, -1):
        midpoint = 15.0 + 10.0 * index
        candidate = backend.full(
            backend.shape(degrees), values[index],
            dtype=execution.float_dtype, device=execution.device,
        )
        result = backend.where(degrees < midpoint - 1e-10, candidate, result)
    return result


def los_probability(
    elevation_rad: Any, environment: str, execution: ExecutionContext
) -> Any:
    """Return nearest-elevation LOS probability from Table 6.6.1-1.

    Exact midpoints select the higher reference elevation.
    """

    document = load_resource("los_probability")
    if environment not in {"dense_urban", "urban", "suburban", "rural"}:
        raise ValueError("unsupported NTN environment")
    return _nearest_table_value(elevation_rad, list(document[environment]), execution)


def sample_los(
    geometry: NTNLinkGeometry,
    topology: NTNTopologyBatch,
    environment: str,
    random_stream: RandomStream[Any],
    execution: ExecutionContext,
) -> Any:
    """Sample one LOS state per link and hold it for the request window."""

    if topology.los_mask is not None:
        return topology.los_mask
    probability = los_probability(geometry.elevation_rad[..., 0], environment, execution)
    draws = random_stream.fork("wireless.tr38811.los").uniform(
        geometry.shape[:3], dtype=execution.float_dtype
    )
    return draws < probability


def free_space_path_loss_db(
    slant_range_m: Any, carrier_frequency_hz: float, execution: ExecutionContext
) -> Any:
    """Return free-space path loss for range in metres and frequency in hertz."""

    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive")
    backend = execution.array_backend
    return 20.0 * backend.log10(slant_range_m) + 20.0 * __import__("math").log10(
        carrier_frequency_hz
    ) - 147.55


def sample_large_scale(
    geometry: NTNLinkGeometry,
    topology: NTNTopologyBatch,
    los_mask: Any,
    config: TR38811DeploymentConfig,
    random_stream: RandomStream[Any],
    execution: ExecutionContext,
    metadata: RealizationMetadata,
    *,
    atmospheric_gas_db: float = 0.0,
    rain_db: float = 0.0,
    cloud_db: float = 0.0,
    scintillation_db: float = 0.0,
) -> NTNLargeScaleComponents:
    """Sample and compose TR 38.811 large-scale attenuation components."""

    backend = execution.array_backend
    document = load_resource("large_scale")
    band = "s" if config.carrier_frequency_hz < 6e9 else "ka"
    table = document["environments"][config.environment][band]
    elevation = geometry.elevation_rad
    los_time = backend.broadcast_to(los_mask[..., None], geometry.shape)
    los_sigma = _nearest_table_value(elevation, list(table["los_sigma"]), execution)
    nlos_sigma = _nearest_table_value(elevation, list(table["nlos_sigma"]), execution)
    sigma = backend.where(los_time, los_sigma, nlos_sigma)
    clutter_table = _nearest_table_value(elevation, list(table["nlos_clutter"]), execution)
    zero = backend.zeros(geometry.shape, dtype=execution.float_dtype, device=execution.device)
    clutter = backend.where(los_time, zero, clutter_table) if config.apply_clutter_loss else zero

    shadow_unit = random_stream.fork("wireless.tr38811.shadow").normal(
        geometry.shape[:3], dtype=execution.float_dtype
    )
    shadow = backend.broadcast_to(shadow_unit[..., None], geometry.shape) * sigma
    if not config.apply_shadow_fading:
        shadow = zero
    indoor = backend.broadcast_to(topology.rx_indoor[:, :, None, None], geometry.shape)
    building = backend.where(indoor, zero + 20.0, zero) if config.apply_o2i_loss else zero
    gas = zero + atmospheric_gas_db if config.apply_atmospheric_loss else zero
    rain = zero + rain_db if config.apply_atmospheric_loss else zero
    cloud = zero + cloud_db if config.apply_atmospheric_loss else zero
    scintillation = zero + scintillation_db if config.apply_scintillation else zero
    free_space = free_space_path_loss_db(
        geometry.slant_range_m, config.carrier_frequency_hz, execution
    )
    total = free_space + clutter + shadow + building + gas + rain + cloud + scintillation
    return NTNLargeScaleComponents(
        free_space, clutter, shadow, building, gas, rain, cloud, scintillation, total, metadata
    )


__all__ = [
    "free_space_path_loss_db", "los_probability", "sample_large_scale", "sample_los",
]
