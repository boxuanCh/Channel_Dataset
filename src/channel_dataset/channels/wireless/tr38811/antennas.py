"""Satellite/HAPS antenna patterns used by TR 38.811."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.channels.wireless.tr38811.config import NTNAntennaConfig
from channel_dataset.core import ExecutionContext


def bessel_j1(value: Any, execution: ExecutionContext) -> Any:
    """Evaluate the first-kind Bessel function :math:`J_1`.

    A convergent power series is used through magnitude 12 and a leading
    asymptotic expansion outside that interval. The implementation is native
    to the selected backend and preserves PyTorch automatic differentiation.
    """

    backend = execution.array_backend
    x = backend.array(value, dtype=execution.float_dtype, device=execution.device)
    half = 0.5 * x
    term = half
    series = term
    squared_quarter = 0.25 * x * x
    for order in range(1, 40):
        term = term * (-squared_quarter / (order * (order + 1)))
        series = series + term
    magnitude = backend.abs(x)
    safe_magnitude = backend.where(
        magnitude > 0.0,
        magnitude,
        backend.ones(backend.shape(magnitude), dtype=execution.float_dtype, device=execution.device),
    )
    asymptotic = backend.sqrt(2.0 / (pi * safe_magnitude)) * backend.cos(
        safe_magnitude - 3.0 * pi / 4.0
    )
    asymptotic = backend.where(x < 0.0, -asymptotic, asymptotic)
    return backend.where(magnitude <= 12.0, series, asymptotic)


def circular_aperture_power_gain(
    off_axis_angle_rad: Any,
    config: NTNAntennaConfig,
    execution: ExecutionContext,
) -> Any:
    """Return normalized circular-aperture power gain in linear units."""

    backend = execution.array_backend
    angle = backend.array(
        off_axis_angle_rad, dtype=execution.float_dtype, device=execution.device
    )
    argument = 2.0 * pi * config.aperture_radius_wavelengths * backend.sin(angle)
    magnitude = backend.abs(argument)
    one = backend.ones(backend.shape(argument), dtype=execution.float_dtype, device=execution.device)
    safe_argument = backend.where(magnitude < 1e-7, one, argument)
    field = 2.0 * bessel_j1(argument, execution) / safe_argument
    field = backend.where(magnitude < 1e-7, one, field)
    return field * field


def circular_aperture_gain_db(
    off_axis_angle_rad: Any,
    config: NTNAntennaConfig,
    execution: ExecutionContext,
) -> Any:
    """Return absolute circular-aperture antenna gain in dBi."""

    backend = execution.array_backend
    power = circular_aperture_power_gain(off_axis_angle_rad, config, execution)
    return config.maximum_gain_dbi + 10.0 * backend.log10(backend.clip(power, 1e-12, 1.0))


__all__ = ["bessel_j1", "circular_aperture_gain_db", "circular_aperture_power_gain"]
