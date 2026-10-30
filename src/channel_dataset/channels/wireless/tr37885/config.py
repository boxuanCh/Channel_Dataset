"""Strict configuration objects for the TR 37.885 urban V2X models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from math import isfinite
from typing import Any

STANDARD_VERSION = "15.3.0"
STANDARD_CARRIER_HZ = 6.0e9


def _finite(value: object, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TR37885UrbanGridConfig:
    """Urban-grid layout and simulation-boundary configuration.

    ``standard`` mode fixes all dimensions to Annex A. Controlled variation
    mode permits a larger integer number of reference-size grid cells.
    """

    mode: str = "standard"
    standard_version: str = STANDARD_VERSION
    cell_width_m: float = 433.0
    cell_height_m: float = 250.0
    cells_x: int = 3
    cells_y: int = 3
    lane_width_m: float = 3.5
    sidewalk_width_m: float = 3.0
    lanes_per_direction: int = 2
    boundary_policy: str = "exit"

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "controlled_variation"}:
            raise ValueError("mode must be 'standard' or 'controlled_variation'")
        if self.standard_version != STANDARD_VERSION:
            raise ValueError(f"standard_version must be {STANDARD_VERSION!r}")
        for name in ("cell_width_m", "cell_height_m", "lane_width_m", "sidewalk_width_m"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("cells_x", "cells_y", "lanes_per_direction"):
            _positive_integer(getattr(self, name), name)
        if self.boundary_policy not in {"exit", "respawn", "periodic"}:
            raise ValueError("invalid boundary_policy")
        if self.mode == "standard":
            expected = {
                "cell_width_m": 433.0,
                "cell_height_m": 250.0,
                "cells_x": 3,
                "cells_y": 3,
                "lane_width_m": 3.5,
                "sidewalk_width_m": 3.0,
                "lanes_per_direction": 2,
                "boundary_policy": "exit",
            }
            changed = [name for name, value in expected.items() if getattr(self, name) != value]
            if changed:
                raise ValueError(f"standard mode fixes grid fields: {sorted(changed)}")

    @property
    def street_width_m(self) -> float:
        """Return the complete four-lane street plus both sidewalks."""

        return 2.0 * self.lanes_per_direction * self.lane_width_m + 2.0 * self.sidewalk_width_m

    @property
    def width_m(self) -> float:
        """Return the simulation width."""

        return self.cells_x * self.cell_width_m

    @property
    def height_m(self) -> float:
        """Return the simulation height."""

        return self.cells_y * self.cell_height_m


@dataclass(frozen=True, slots=True)
class TR37885TrafficConfig:
    """Vehicle population and mobility controls for urban Options A/B."""

    option: str = "A"
    mode: str = "standard"
    num_vehicles: int = 32
    update_interval_s: float = 0.1
    speed_multiplier: float = 1.0
    vehicle_type_probabilities: tuple[float, float, float] | None = None
    turn_probabilities: tuple[float, float, float] = (0.5, 0.25, 0.25)
    density_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.option not in {"A", "B"}:
            raise ValueError("option must be 'A' or 'B'")
        if self.mode not in {"standard", "controlled_variation"}:
            raise ValueError("mode must be 'standard' or 'controlled_variation'")
        _positive_integer(self.num_vehicles, "num_vehicles")
        for name in ("update_interval_s", "speed_multiplier", "density_multiplier"):
            _finite(getattr(self, name), name, positive=True)
        probabilities = self.vehicle_type_probabilities
        if probabilities is not None:
            _probabilities(probabilities, "vehicle_type_probabilities")
        _probabilities(self.turn_probabilities, "turn_probabilities")
        if self.mode == "standard":
            if self.update_interval_s != 0.1 or self.speed_multiplier != 1.0:
                raise ValueError("standard mode fixes update_interval_s and speed_multiplier")
            if self.density_multiplier != 1.0:
                raise ValueError("standard mode fixes density_multiplier")
            expected_types = None
            if probabilities != expected_types:
                raise ValueError("standard mode derives vehicle type probabilities from option")
            if self.turn_probabilities != (0.5, 0.25, 0.25):
                raise ValueError("standard mode fixes turn probabilities")

    @property
    def resolved_vehicle_type_probabilities(self) -> tuple[float, float, float]:
        """Return probabilities for vehicle types 1, 2, and 3."""

        if self.vehicle_type_probabilities is not None:
            return self.vehicle_type_probabilities
        return (0.0, 1.0, 0.0) if self.option == "A" else (0.2, 0.6, 0.2)


def _probabilities(values: tuple[float, ...], name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three values")
    for value in values:
        _finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} values must be nonnegative")
    if abs(sum(values) - 1.0) > 1.0e-12:
        raise ValueError(f"{name} must sum to one")


@dataclass(frozen=True, slots=True)
class TR37885UrbanSystemConfig:
    """Stochastic system-level urban V2V channel configuration."""

    standard_version: str = STANDARD_VERSION
    carrier_frequency_hz: float = STANDARD_CARRIER_HZ
    bandwidth_hz: float = 20.0e6
    num_rx_antennas: int = 1
    num_tx_antennas: int = 1
    apply_path_loss: bool = True
    apply_shadow_fading: bool = True
    apply_vehicle_blockage: bool = True
    moving_scatterers: bool = True

    def __post_init__(self) -> None:
        _validate_channel_config(self)


@dataclass(frozen=True, slots=True)
class TR37885UrbanCDLConfig:
    """Fixed-profile urban V2V CDL channel configuration."""

    standard_version: str = STANDARD_VERSION
    carrier_frequency_hz: float = STANDARD_CARRIER_HZ
    bandwidth_hz: float = 20.0e6
    num_rx_antennas: int = 1
    num_tx_antennas: int = 1
    apply_path_loss: bool = True
    apply_shadow_fading: bool = True
    apply_vehicle_blockage: bool = True
    moving_scatterers: bool = True

    def __post_init__(self) -> None:
        _validate_channel_config(self)


def _validate_channel_config(config: Any) -> None:
    if config.standard_version != STANDARD_VERSION:
        raise ValueError(f"standard_version must be {STANDARD_VERSION!r}")
    _finite(config.carrier_frequency_hz, "carrier_frequency_hz", positive=True)
    _finite(config.bandwidth_hz, "bandwidth_hz", positive=True)
    for name in ("num_rx_antennas", "num_tx_antennas"):
        _positive_integer(getattr(config, name), name)
    for name in (
        "apply_path_loss", "apply_shadow_fading", "apply_vehicle_blockage",
        "moving_scatterers",
    ):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be Boolean")


def _parse(parameters: Mapping[str, Any], cls: type[Any], label: str) -> Any:
    expected = {item.name for item in fields(cls)}
    required = {
        item.name for item in fields(cls)
        if item.default is MISSING and item.default_factory is MISSING
    }
    unknown = set(parameters) - expected
    missing = required - set(parameters)
    if unknown or missing:
        raise ValueError(f"invalid {label}; missing={sorted(missing)}, unknown={sorted(unknown)}")
    values = dict(parameters)
    for name in ("vehicle_type_probabilities", "turn_probabilities"):
        if name in values and isinstance(values[name], (list, tuple)):
            values[name] = tuple(values[name])
    return cls(**values)


def parse_urban_system_config(parameters: Mapping[str, Any]) -> TR37885UrbanSystemConfig:
    """Parse strict stochastic urban-channel parameters."""

    return _parse(parameters, TR37885UrbanSystemConfig, "TR 37.885 urban system parameters")


def parse_urban_cdl_config(parameters: Mapping[str, Any]) -> TR37885UrbanCDLConfig:
    """Parse strict urban-CDL parameters."""

    return _parse(parameters, TR37885UrbanCDLConfig, "TR 37.885 urban CDL parameters")


__all__ = [
    "STANDARD_CARRIER_HZ", "STANDARD_VERSION", "TR37885TrafficConfig",
    "TR37885UrbanCDLConfig", "TR37885UrbanGridConfig", "TR37885UrbanSystemConfig",
    "parse_urban_cdl_config", "parse_urban_system_config",
]
