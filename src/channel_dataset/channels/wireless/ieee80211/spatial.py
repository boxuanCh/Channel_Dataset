"""TGn power-angular spectra and per-tap spatial covariance."""

from __future__ import annotations

from cmath import exp
from math import cos, pi, sqrt
from typing import Any, Literal

from channel_dataset.channels.wireless.ieee80211.profiles import IEEE80211Profile
from channel_dataset.core import ExecutionContext


def _wrapped_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


def laplacian_ula_covariance(
    num_antennas: int,
    spacing_wavelengths: float,
    mean_angle_rad: float,
    angular_spread_rad: float,
    execution: ExecutionContext,
    *,
    quadrature_order: int = 256,
) -> Any:
    r"""Integrate a truncated Laplacian PAS for a uniform linear array.

    The correlation between elements ``m`` and ``n`` is the normalized
    numerical integral of
    :math:`p(\phi)e^{j2\pi d(m-n)\cos(\phi)}` over ``[-pi, pi)``.
    This implements IEEE 802.11-03/940r4, sections 3 and 4.2.
    """

    if isinstance(num_antennas, bool) or not isinstance(num_antennas, int):
        raise TypeError("num_antennas must be an integer")
    if num_antennas <= 0 or spacing_wavelengths <= 0 or angular_spread_rad <= 0:
        raise ValueError("antenna count, spacing, and angular spread must be positive")
    if quadrature_order < 32:
        raise ValueError("quadrature_order must be at least 32")
    angles = [-pi + (index + 0.5) * 2.0 * pi / quadrature_order for index in range(quadrature_order)]
    weights = [
        exp(-sqrt(2.0) * abs(_wrapped_angle(angle - mean_angle_rad)) / angular_spread_rad).real
        for angle in angles
    ]
    scale = sum(weights)
    matrix = []
    for row in range(num_antennas):
        values = []
        for column in range(num_antennas):
            separation = spacing_wavelengths * (row - column)
            value = sum(
                weight * exp(1j * 2.0 * pi * separation * cos(angle))
                for angle, weight in zip(angles, weights, strict=True)
            ) / scale
            values.append(value)
        matrix.append(values)
    return execution.array_backend.array(
        matrix, dtype=execution.complex_dtype, device=execution.device
    )


def tap_spatial_covariances(
    profile: IEEE80211Profile,
    num_antennas: int,
    spacing_wavelengths: float,
    side: Literal["rx", "tx"],
    execution: ExecutionContext,
    *,
    quadrature_order: int = 256,
) -> Any:
    """Return power-weighted ULA covariance for every resolved profile tap.

    Returns
    -------
    backend array
        Complex array with shape ``[tap, antenna, antenna]``.
    """

    if side not in {"rx", "tx"}:
        raise ValueError("side must be 'rx' or 'tx'")
    backend = execution.array_backend
    matrices = []
    for tap_index, tap_power in enumerate(profile.raw_tap_powers_linear, start=1):
        weighted = None
        for cluster in profile.clusters:
            if tap_index not in cluster.tap_indices:
                continue
            local = cluster.tap_indices.index(tap_index)
            contribution = 10.0 ** (cluster.powers_db[local] / 10.0) / tap_power
            mean = cluster.mean_aoa_rad if side == "rx" else cluster.mean_aod_rad
            spread = (
                cluster.rx_angular_spread_rad if side == "rx" else cluster.tx_angular_spread_rad
            )
            covariance = laplacian_ula_covariance(
                num_antennas, spacing_wavelengths, mean, spread, execution,
                quadrature_order=quadrature_order,
            )
            weighted = contribution * covariance if weighted is None else weighted + contribution * covariance
        assert weighted is not None
        matrices.append(weighted)
    return backend.stack(tuple(matrices), axis=0)


__all__ = ["laplacian_ula_covariance", "tap_spatial_covariances"]
