"""TR 37.885 urban V2V path loss, shadowing, and vehicle blockage."""

from __future__ import annotations

from math import log10

from channel_dataset.channels.wireless.tr37885.types import LOS, NLOS, NLOSV


def urban_path_loss_db(distance_3d_m: float, carrier_frequency_hz: float, condition: int) -> float:
    """Evaluate the urban V2V path-loss equations in clause 6.2.1."""

    if distance_3d_m <= 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("distance and carrier frequency must be positive")
    frequency_ghz = carrier_frequency_hz / 1.0e9
    if condition in {LOS, NLOSV}:
        return 38.77 + 16.7 * log10(distance_3d_m) + 18.2 * log10(frequency_ghz)
    if condition == NLOS:
        return 36.85 + 30.0 * log10(distance_3d_m) + 18.9 * log10(frequency_ghz)
    raise ValueError("path loss requires LOS, NLOSv, or NLOS")


def vehicle_blockage_parameters_db(
    distance_m: float,
    tx_antenna_height_m: float,
    rx_antenna_height_m: float,
    blocker_height_m: float,
) -> tuple[float, float]:
    """Return Gaussian-dB mean and standard deviation for NLOSv loss."""

    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if min(tx_antenna_height_m, rx_antenna_height_m) > blocker_height_m:
        return 0.0, 0.0
    distance_term = max(0.0, 15.0 * log10(distance_m) - 41.0)
    if max(tx_antenna_height_m, rx_antenna_height_m) < blocker_height_m:
        return 9.0 + distance_term, 4.5
    return 5.0 + distance_term, 4.0


__all__ = ["urban_path_loss_db", "vehicle_blockage_parameters_db"]
