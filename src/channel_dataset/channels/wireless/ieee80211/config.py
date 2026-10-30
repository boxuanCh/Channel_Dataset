"""Strict typed configuration for IEEE 802.11 WLAN channel models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from typing import Any


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TGnConfig:
    """IEEE 802.11-03/940r4 TGn channel configuration.

    Parameters are physical except ``num_sinusoids`` and
    ``spatial_quadrature_order``, which control numerical approximation.
    Large-scale modes other than ``"none"`` require topology at realization.
    """

    profile: str
    standard_version: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    environmental_speed_mps: float
    num_sinusoids: int
    spatial_quadrature_order: int
    rx_antenna_spacing_wavelengths: float
    tx_antenna_spacing_wavelengths: float
    line_of_sight: bool
    large_scale_mode: str
    fluorescent_effect: bool
    moving_vehicle_effect: bool
    normalize_profile_power: bool

    def __post_init__(self) -> None:
        if self.profile not in set("ABCDEF"):
            raise ValueError("profile must be A through F")
        if self.standard_version != "03/940r4":
            raise ValueError("standard_version must be '03/940r4'")
        _positive(self.carrier_frequency_hz, "carrier_frequency_hz")
        if not 2e9 <= self.carrier_frequency_hz <= 6e9:
            raise ValueError("TGn carrier_frequency_hz must be between 2 and 6 GHz")
        _positive(self.bandwidth_hz, "bandwidth_hz")
        if self.bandwidth_hz > 40e6:
            raise ValueError("TGn bandwidth_hz must not exceed 40 MHz")
        for name in (
            "num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas",
            "num_sinusoids", "spatial_quadrature_order",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_rx_antennas > 4 or self.num_tx_antennas > 4:
            raise ValueError("TGn supports at most 4-by-4 MIMO")
        if self.spatial_quadrature_order < 32:
            raise ValueError("spatial_quadrature_order must be at least 32")
        _nonnegative(self.environmental_speed_mps, "environmental_speed_mps")
        _positive(self.rx_antenna_spacing_wavelengths, "rx_antenna_spacing_wavelengths")
        _positive(self.tx_antenna_spacing_wavelengths, "tx_antenna_spacing_wavelengths")
        if self.large_scale_mode not in {"none", "path_loss", "path_loss_shadowing"}:
            raise ValueError("large_scale_mode must be 'none', 'path_loss', or 'path_loss_shadowing'")
        if self.fluorescent_effect and self.profile not in {"D", "E"}:
            raise ValueError("fluorescent_effect is defined only for profiles D and E")
        if self.moving_vehicle_effect and self.profile != "F":
            raise ValueError("moving_vehicle_effect is defined only for profile F")


@dataclass(frozen=True, slots=True)
class TGacConfig:
    """IEEE 802.11-09/0308r12 TGac extension configuration."""

    profile: str
    standard_version: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    environmental_speed_mps: float
    num_sinusoids: int
    spatial_quadrature_order: int
    rx_antenna_spacing_wavelengths: float
    tx_antenna_spacing_wavelengths: float
    line_of_sight: bool
    large_scale_mode: str
    fluorescent_effect: bool
    moving_vehicle_effect: bool
    normalize_profile_power: bool
    num_users: int
    direction: str

    def __post_init__(self) -> None:
        _validate_extended(self, "09/0308r12", {20e6, 40e6, 80e6, 160e6}, 8)
        if not 4.9e9 <= self.carrier_frequency_hz <= 6e9:
            raise ValueError("TGac carrier_frequency_hz must be in the 5 GHz WLAN range")
        if self.direction not in {"downlink", "uplink"}:
            raise ValueError("direction must be 'downlink' or 'uplink'")
        if isinstance(self.num_users, bool) or not isinstance(self.num_users, int):
            raise TypeError("num_users must be an integer")
        if self.num_users <= 0:
            raise ValueError("num_users must be positive")
        expected = self.num_rx if self.direction == "downlink" else self.num_tx
        if self.num_users != expected:
            raise ValueError("num_users must match receiver links downlink or transmitter links uplink")


@dataclass(frozen=True, slots=True)
class TGahConfig:
    """IEEE 802.11-11/0968r4 sub-1-GHz TGah configuration."""

    profile: str
    standard_version: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    environmental_speed_mps: float
    num_sinusoids: int
    spatial_quadrature_order: int
    rx_antenna_spacing_wavelengths: float
    tx_antenna_spacing_wavelengths: float
    line_of_sight: bool
    large_scale_mode: str
    fluorescent_effect: bool
    moving_vehicle_effect: bool
    normalize_profile_power: bool
    link_type: str
    deployment: str

    def __post_init__(self) -> None:
        _validate_extended(self, "11/0968r4", {1e6, 2e6, 4e6, 8e6, 16e6}, 8)
        if not 700e6 <= self.carrier_frequency_hz < 1e9:
            raise ValueError("TGah requires a sub-1-GHz carrier from 700 MHz to 1 GHz")
        if self.link_type not in {"ap_to_station", "station_to_station"}:
            raise ValueError("invalid TGah link_type")
        if self.deployment not in {"indoor", "outdoor_macro", "outdoor_pico", "outdoor_hotspot"}:
            raise ValueError("invalid TGah deployment")


@dataclass(frozen=True, slots=True)
class TGaxConfig:
    """IEEE 802.11-14/0882r4 claimed indoor/outdoor TGax subset."""

    profile: str
    standard_version: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    environmental_speed_mps: float
    num_sinusoids: int
    spatial_quadrature_order: int
    rx_antenna_spacing_wavelengths: float
    tx_antenna_spacing_wavelengths: float
    line_of_sight: bool
    large_scale_mode: str
    fluorescent_effect: bool
    moving_vehicle_effect: bool
    normalize_profile_power: bool
    scenario: str
    wall_count: int
    floor_count: int
    fast_object_effect: bool
    street_width_m: float
    average_building_height_m: float

    def __post_init__(self) -> None:
        _validate_extended(self, "14/0882r4", {20e6, 40e6, 80e6, 160e6}, 8)
        if not 2e9 <= self.carrier_frequency_hz <= 6e9:
            raise ValueError("TGax carrier_frequency_hz must be between 2 and 6 GHz")
        if self.scenario not in {"indoor_residential", "indoor_enterprise", "outdoor_umi", "outdoor_uma"}:
            raise ValueError("invalid TGax scenario")
        for name in ("wall_count", "floor_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.scenario.startswith("outdoor") and (self.wall_count or self.floor_count):
            raise ValueError("outdoor TGax scenarios do not accept wall/floor penetration")
        _positive(self.street_width_m, "street_width_m")
        _positive(self.average_building_height_m, "average_building_height_m")


def _validate_extended(config: Any, version: str, bandwidths: set[float], max_antennas: int) -> None:
    if config.profile not in set("ABCDEF"):
        raise ValueError("profile must be A through F")
    if config.standard_version != version:
        raise ValueError(f"standard_version must be {version!r}")
    _positive(config.carrier_frequency_hz, "carrier_frequency_hz")
    _positive(config.bandwidth_hz, "bandwidth_hz")
    if config.bandwidth_hz not in bandwidths:
        raise ValueError(f"unsupported channel bandwidth: {config.bandwidth_hz}")
    for name in (
        "num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas",
        "num_sinusoids", "spatial_quadrature_order",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.num_rx_antennas > max_antennas or config.num_tx_antennas > max_antennas:
        raise ValueError(f"model supports at most {max_antennas}-by-{max_antennas} MIMO")
    if config.spatial_quadrature_order < 32:
        raise ValueError("spatial_quadrature_order must be at least 32")
    _nonnegative(config.environmental_speed_mps, "environmental_speed_mps")
    _positive(config.rx_antenna_spacing_wavelengths, "rx_antenna_spacing_wavelengths")
    _positive(config.tx_antenna_spacing_wavelengths, "tx_antenna_spacing_wavelengths")
    if config.large_scale_mode not in {"none", "path_loss", "path_loss_shadowing"}:
        raise ValueError("invalid large_scale_mode")
    if config.fluorescent_effect and config.profile not in {"D", "E"}:
        raise ValueError("fluorescent_effect is defined only for profiles D and E")
    if config.moving_vehicle_effect and config.profile != "F":
        raise ValueError("moving_vehicle_effect is defined only for profile F")


def _parse(parameters: Mapping[str, Any], cls: type[Any], label: str) -> Any:
    expected = {field.name for field in fields(cls)}
    unknown = set(parameters) - expected
    missing = expected - set(parameters)
    if unknown or missing:
        raise ValueError(f"invalid {label} parameters; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return cls(**dict(parameters))


def parse_tgn_config(parameters: Mapping[str, Any]) -> TGnConfig:
    """Parse a strict :class:`TGnConfig` mapping."""
    return _parse(parameters, TGnConfig, "TGn")


def parse_tgac_config(parameters: Mapping[str, Any]) -> TGacConfig:
    """Parse a strict :class:`TGacConfig` mapping."""
    return _parse(parameters, TGacConfig, "TGac")


def parse_tgah_config(parameters: Mapping[str, Any]) -> TGahConfig:
    """Parse a strict :class:`TGahConfig` mapping."""
    return _parse(parameters, TGahConfig, "TGah")


def parse_tgax_config(parameters: Mapping[str, Any]) -> TGaxConfig:
    """Parse a strict :class:`TGaxConfig` mapping."""
    return _parse(parameters, TGaxConfig, "TGax")


__all__ = [
    "TGacConfig", "TGahConfig", "TGaxConfig", "TGnConfig", "parse_tgac_config",
    "parse_tgah_config", "parse_tgax_config", "parse_tgn_config",
]
