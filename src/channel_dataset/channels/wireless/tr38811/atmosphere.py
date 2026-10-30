"""Composable Earth-space atmospheric attenuation inputs."""

from __future__ import annotations

from typing import Any, Protocol

from channel_dataset.channels.wireless.tr38811.config import NTNAtmosphereConfig
from channel_dataset.core import ExecutionContext


class NTNAtmosphereProvider(Protocol):
    """Optional provider for site- and time-specific attenuation inputs."""

    def zenith_gas_attenuation_db(
        self, carrier_frequency_hz: float, atmosphere: NTNAtmosphereConfig
    ) -> float:
        """Return P.676 zenith gaseous attenuation in decibels."""

        ...

    def rain_attenuation_db(
        self, carrier_frequency_hz: float, elevation_rad: Any,
        atmosphere: NTNAtmosphereConfig, execution: ExecutionContext,
    ) -> Any:
        """Return path rain attenuation on the execution backend."""

        ...

    def cloud_attenuation_db(
        self, carrier_frequency_hz: float, elevation_rad: Any,
        atmosphere: NTNAtmosphereConfig, execution: ExecutionContext,
    ) -> Any:
        """Return path cloud attenuation on the execution backend."""

        ...


def slant_gaseous_attenuation_db(
    zenith_attenuation_db: float,
    elevation_rad: Any,
    execution: ExecutionContext,
) -> Any:
    """Scale P.676 zenith attenuation by the TR 38.811 elevation mapping."""

    if zenith_attenuation_db < 0.0:
        raise ValueError("zenith_attenuation_db must be nonnegative")
    backend = execution.array_backend
    elevation = backend.array(
        elevation_rad, dtype=execution.float_dtype, device=execution.device
    )
    sine = backend.sin(elevation)
    if not backend.all_nonnegative(sine):
        raise ValueError("gaseous attenuation requires a visible positive-elevation link")
    return zenith_attenuation_db / backend.clip(sine, 1e-6, 1.0)


def should_compute_gaseous_attenuation(
    carrier_frequency_hz: float, elevation_rad: Any, execution: ExecutionContext
) -> bool:
    """Return the clause-6.6.4 applicability decision."""

    if carrier_frequency_hz >= 10e9:
        return True
    if carrier_frequency_hz <= 1e9:
        return False
    backend = execution.array_backend
    threshold = 10.0 * __import__("math").pi / 180.0
    return bool(backend.to_scalar(backend.min(elevation_rad)) < threshold)


__all__ = [
    "NTNAtmosphereProvider", "should_compute_gaseous_attenuation",
    "slant_gaseous_attenuation_db",
]
