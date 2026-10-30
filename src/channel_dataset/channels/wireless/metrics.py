"""Backend-native propagation and MIMO channel metrics."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.geometric_cdl.types import DynamicCIRBatch
from channel_dataset.channels.wireless.types import CIRBatch, OFDMChannelResponse, RayBatch


def rms_delay_spread(cir: CIRBatch | DynamicCIRBatch) -> Any:
    """Compute RMS delay spread for every batch/receiver/transmitter link.

    Returns
    -------
    backend array
        Seconds on axes ``[batch, rx, tx]``.
    """

    backend = get_backend(cir.metadata.backend)
    if isinstance(cir, DynamicCIRBatch):
        power = backend.mean(backend.abs(cir.coefficients) ** 2, axis=(2, 4))
        # [batch, rx, tx, path, time]
        if cir.path_mask is not None:
            power = power * backend.astype(
                cir.path_mask,
                "float32" if cir.metadata.precision == "single" else "float64",
            )
        total = backend.sum(power, axis=-2) + 1.0e-30
        mean_delay = backend.sum(power * cir.delays_s, axis=-2) / total
        variance = backend.sum(
            power * (cir.delays_s - mean_delay[..., None, :]) ** 2, axis=-2
        ) / total
        return backend.sqrt(variance)
    power = backend.mean(backend.abs(cir.coefficients) ** 2, axis=(2, 4, 6))
    if cir.path_mask is not None:
        power = power * backend.astype(
            cir.path_mask,
            "float32" if cir.metadata.precision == "single" else "float64",
        )
    total = backend.sum(power, axis=-1) + 1.0e-30
    mean_delay = backend.sum(power * cir.delays_s, axis=-1) / total
    variance = backend.sum(
        power * (cir.delays_s - mean_delay[..., None]) ** 2, axis=-1
    ) / total
    return backend.sqrt(variance)


def circular_angular_spread(
    rays: RayBatch, *, angle: str = "azimuth_arrival_rad"
) -> Any:
    """Compute power-weighted circular angular spread per link.

    Parameters
    ----------
    rays : RayBatch
        Ray angles and cluster powers.
    angle : str, default="azimuth_arrival_rad"
        One of the four per-ray angle field names.
    """

    allowed = {
        "azimuth_arrival_rad",
        "zenith_arrival_rad",
        "azimuth_departure_rad",
        "zenith_departure_rad",
    }
    if angle not in allowed:
        raise ValueError(f"angle must be one of {sorted(allowed)}")
    backend = get_backend(rays.metadata.backend)
    values = getattr(rays, angle)
    ray_count = rays.shape[-1]
    weight = rays.powers_linear[..., None] / ray_count
    if rays.ray_mask is not None:
        weight = weight * backend.astype(
            rays.ray_mask,
            "float32" if rays.metadata.precision == "single" else "float64",
        )
    real = backend.sum(weight * backend.cos(values), axis=(-2, -1))
    imag = backend.sum(weight * backend.sin(values), axis=(-2, -1))
    resultant = backend.sqrt(real * real + imag * imag)
    resultant = (resultant + 1.0 - backend.abs(resultant - 1.0)) * 0.5
    return backend.sqrt(-2.0 * backend.log(resultant + 1.0e-30))


def mimo_singular_values(response: OFDMChannelResponse) -> Any:
    """Return MIMO singular values per link, symbol, and subcarrier.

    Returns
    -------
    backend array
        Axes ``[batch, rx, tx, symbol, subcarrier, singular_value]``.
    """

    backend = get_backend(response.metadata.backend)
    matrices = backend.transpose(response.coefficients, (0, 1, 3, 5, 6, 2, 4))
    return backend.svdvals(matrices)


def frequency_selectivity(response: OFDMChannelResponse) -> Any:
    """Return normalized RMS power variation across subcarriers per link."""

    backend = get_backend(response.metadata.backend)
    power = backend.mean(backend.abs(response.coefficients) ** 2, axis=(2, 4, 5))
    mean = backend.mean(power, axis=-1)
    variance = backend.mean((power - mean[..., None]) ** 2, axis=-1)
    return backend.sqrt(variance) / (mean + 1.0e-30)


__all__ = [
    "circular_angular_spread",
    "frequency_selectivity",
    "mimo_singular_values",
    "rms_delay_spread",
]
