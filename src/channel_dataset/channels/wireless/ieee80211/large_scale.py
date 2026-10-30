"""IEEE 802.11 path-loss, shadowing, and penetration components."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, pi
from typing import Any

from channel_dataset.backends import ArrayBackend, RandomStream
from channel_dataset.channels.wireless.ieee80211.profiles import (
    load_tgah_path_loss_profile,
    load_tgax_indoor_profile,
    load_tgax_outdoor_profile,
)
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS


@dataclass(frozen=True, slots=True)
class WLANPathLossResult:
    """Resolved large-scale loss components, all in decibels."""

    path_loss_db: Any
    shadow_fading_db: Any
    penetration_loss_db: Any


def free_space_path_loss_db(distance_m: Any, carrier_frequency_hz: float, backend: ArrayBackend[Any]) -> Any:
    """Return free-space path loss for positive distances in metres."""

    if float(backend.to_scalar(backend.min(distance_m))) <= 0.0:
        raise ValueError("link distances must be positive")
    wavelength_m = SPEED_OF_LIGHT_MPS / carrier_frequency_hz
    return 20.0 * backend.log10(4.0 * pi * distance_m / wavelength_m)


def tgn_breakpoint_path_loss_db(
    distance_m: Any,
    carrier_frequency_hz: float,
    breakpoint_distance_m: float,
    backend: ArrayBackend[Any],
) -> Any:
    """Evaluate continuous TGn free-space/breakpoint path loss.

    IEEE 802.11-03/940r4 equation (1) uses free-space slope up to the
    breakpoint and 35 dB per distance decade after it.
    """

    if float(backend.to_scalar(backend.min(distance_m))) <= 0.0:
        raise ValueError("link distances must be positive")
    # min(d, breakpoint) without adding a backend protocol operation.
    ratio = distance_m / breakpoint_distance_m
    below = (ratio + 1.0 - backend.abs(ratio - 1.0)) * 0.5
    above = (ratio - 1.0 + backend.abs(ratio - 1.0)) * 0.5 + 1.0
    breakpoint = backend.full(
        backend.shape(distance_m), breakpoint_distance_m,
        dtype=backend.dtype(distance_m), device=backend.device(distance_m),
    )
    return free_space_path_loss_db(breakpoint * below, carrier_frequency_hz, backend) + 35.0 * backend.log10(above)


def tgah_path_loss_db(
    distance_m: Any,
    carrier_frequency_hz: float,
    deployment: str,
    link_type: str,
    backend: ArrayBackend[Any],
) -> Any:
    """Evaluate the supported TGah sub-1-GHz deployment loss subset.

    The reference-distance log-distance parameters are transcribed from the
    scenario equations in IEEE 802.11-11/0968r4. AP-to-station links use the
    scenario exponent; station-to-station links add the documented 3-dB
    endpoint-height allowance used by this implementation subset.
    """

    profile = load_tgah_path_loss_profile()
    exponent = profile.path_loss_exponents[deployment]
    base = free_space_path_loss_db(
        backend.ones(backend.shape(distance_m), dtype=backend.dtype(distance_m), device=backend.device(distance_m)),
        carrier_frequency_hz,
        backend,
    )
    extra = profile.station_to_station_additional_loss_db if link_type == "station_to_station" else 0.0
    return base + 10.0 * exponent * backend.log10(distance_m) + extra


def tgax_path_loss_db(
    distance_m: Any,
    carrier_frequency_hz: float,
    scenario: str,
    line_of_sight: bool,
    backend: ArrayBackend[Any],
    *,
    ap_height_m: Any | None = None,
    station_height_m: Any | None = None,
    street_width_m: float = 20.0,
    average_building_height_m: float = 20.0,
) -> Any:
    """Evaluate TGax indoor and outdoor path loss from 14/0882r4.

    Outdoor LoS applies the height-dependent breakpoint equation. UMi NLoS
    uses the document's explicit hexagonal-layout formula. UMa NLoS uses the
    ITU-R M.2135 expression referenced by the TGax document with explicit
    street width and average building height.
    """

    if scenario.startswith("indoor"):
        breakpoint = load_tgax_indoor_profile().breakpoint_distance_m[scenario]
        return tgn_breakpoint_path_loss_db(distance_m, carrier_frequency_hz, breakpoint, backend)
    if ap_height_m is None or station_height_m is None:
        raise ValueError("outdoor TGax path loss requires AP and station heights")
    if float(backend.to_scalar(backend.min(ap_height_m))) <= 1.0:
        raise ValueError("outdoor TGax AP height must exceed 1 m")
    if float(backend.to_scalar(backend.min(station_height_m))) <= 1.0:
        raise ValueError("outdoor TGax station height must exceed 1 m")
    if float(backend.to_scalar(backend.min(distance_m))) < 10.0:
        raise ValueError("outdoor TGax path loss requires distance of at least 10 m")
    profile = load_tgax_outdoor_profile()
    coefficients = profile.scenarios[scenario]
    distance_3d = distance_m
    frequency_ghz = carrier_frequency_hz / 1e9
    before = coefficients["los_before"]
    after = coefficients["los_after"]
    los_before = (
        before["intercept_db"]
        + before["distance_coefficient"] * backend.log10(distance_3d)
        + before["frequency_coefficient"] * log10(frequency_ghz)
    )
    breakpoint = (
        4.0 * (ap_height_m - 1.0) * (station_height_m - 1.0)
        * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    )
    los_after = (
        after["intercept_db"]
        + after["distance_coefficient"] * backend.log10(distance_3d)
        + after["height_coefficient"] * backend.log10(ap_height_m - 1.0)
        + after["height_coefficient"] * backend.log10(station_height_m - 1.0)
        + after["frequency_coefficient"] * log10(frequency_ghz)
    )
    # The branches meet at the breakpoint; the larger branch selects the
    # 22-dB/decade law below it and the 40-dB/decade law above it.
    los = (los_before + los_after + backend.abs(los_before - los_after)) * 0.5
    if line_of_sight:
        return los
    if scenario == "outdoor_umi":
        nlos_parameters = coefficients["nlos"]
        nlos = (
            nlos_parameters["intercept_db"]
            + nlos_parameters["distance_coefficient"] * backend.log10(distance_3d)
            + nlos_parameters["frequency_coefficient"] * log10(frequency_ghz)
        )
    else:
        nlos_parameters = coefficients["nlos"]
        h_ap = ap_height_m
        h_sta = station_height_m
        h_building = average_building_height_m
        nlos = (
            nlos_parameters["intercept_db"]
            + nlos_parameters["street_width_coefficient"] * log10(street_width_m)
            + nlos_parameters["building_height_coefficient"] * log10(h_building)
            - (nlos_parameters["ap_height_base"] - nlos_parameters["building_ratio_coefficient"] * (h_building / h_ap) ** 2) * backend.log10(h_ap)
            + (nlos_parameters["distance_base"] + nlos_parameters["distance_ap_log_coefficient"] * backend.log10(h_ap))
            * (backend.log10(distance_3d) + nlos_parameters["distance_log_offset"])
            + nlos_parameters["frequency_coefficient"] * log10(frequency_ghz)
            - (nlos_parameters["correction_coefficient"] * (log10(nlos_parameters["correction_height_m"]) ** 2) - nlos_parameters["correction_offset"])
            + nlos_parameters["station_height_coefficient"]
            * (h_sta - nlos_parameters["station_reference_height_m"])
        )
    return (los + nlos + backend.abs(los - nlos)) * 0.5


def sample_link_shadowing(
    stream: RandomStream[Any],
    shape: tuple[int, ...],
    standard_deviation_db: float,
    *,
    dtype: str,
) -> Any:
    """Sample zero-mean link shadow fading from an isolated child stream."""

    return stream.normal(shape, 0.0, standard_deviation_db, dtype=dtype)


__all__ = [
    "WLANPathLossResult", "free_space_path_loss_db", "sample_link_shadowing",
    "tgah_path_loss_db", "tgax_path_loss_db", "tgn_breakpoint_path_loss_db",
]
