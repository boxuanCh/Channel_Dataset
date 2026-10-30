"""Strict typed configuration for 3GPP TR 38.811 channel models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from math import isfinite
from typing import Any

STANDARD_VERSION = "15.4.0"
DEPLOYMENTS = frozenset({"D1", "D2", "D3", "D4", "D5"})
ENVIRONMENTS = frozenset({"dense_urban", "urban", "suburban", "rural"})


def _finite(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _boolean(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be Boolean")


def _validate_standard(version: str) -> None:
    if version != STANDARD_VERSION:
        raise ValueError(f"standard_version must be {STANDARD_VERSION!r}")


def _validate_deployment(name: str, environment: str) -> None:
    if name not in DEPLOYMENTS:
        raise ValueError("deployment must be D1 through D5")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {sorted(ENVIRONMENTS)}")


@dataclass(frozen=True, slots=True)
class TR38811DeploymentConfig:
    """Deployment applicability and large-scale propagation controls."""

    deployment: str
    standard_version: str
    environment: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    apply_clutter_loss: bool = True
    apply_shadow_fading: bool = True
    apply_o2i_loss: bool = True
    apply_atmospheric_loss: bool = True
    apply_scintillation: bool = True

    def __post_init__(self) -> None:
        _validate_standard(self.standard_version)
        _validate_deployment(self.deployment, self.environment)
        _finite(self.carrier_frequency_hz, "carrier_frequency_hz", positive=True)
        _finite(self.bandwidth_hz, "bandwidth_hz", positive=True)
        for name in (
            "apply_clutter_loss", "apply_shadow_fading", "apply_o2i_loss",
            "apply_atmospheric_loss", "apply_scintillation",
        ):
            _boolean(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class TR38811TDLConfig:
    """TR 38.811 NTN tapped-delay-line configuration."""

    profile: str
    standard_version: str
    deployment: str
    environment: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    delay_spread_s: float
    num_rx_antennas: int
    num_tx_antennas: int
    num_sinusoids: int = 32
    terminal_speed_mps: float = 0.0
    k_factor_db: float | None = None
    apply_large_scale: bool = True

    def __post_init__(self) -> None:
        if self.profile not in {"NTN-TDL-A", "NTN-TDL-B", "NTN-TDL-C", "NTN-TDL-D"}:
            raise ValueError("profile must be NTN-TDL-A through NTN-TDL-D")
        _validate_standard(self.standard_version)
        _validate_deployment(self.deployment, self.environment)
        for name in ("carrier_frequency_hz", "bandwidth_hz", "delay_spread_s"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("num_rx_antennas", "num_tx_antennas", "num_sinusoids"):
            _positive_integer(getattr(self, name), name)
        _finite(self.terminal_speed_mps, "terminal_speed_mps", nonnegative=True)
        if self.k_factor_db is not None:
            _finite(self.k_factor_db, "k_factor_db")
            if self.profile not in {"NTN-TDL-C", "NTN-TDL-D"}:
                raise ValueError("k_factor_db is supported only for NTN-TDL-C/D")
        _boolean(self.apply_large_scale, "apply_large_scale")


@dataclass(frozen=True, slots=True)
class TR38811CDLConfig:
    """TR 38.811 NTN clustered-delay-line configuration."""

    profile: str
    standard_version: str
    deployment: str
    environment: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    delay_spread_s: float
    desired_asa_degrees: float
    desired_zsa_degrees: float
    tx_array_profile: str
    rx_array_profile: str
    k_factor_db: float | None = None
    faraday_rotation_rad: float = 0.0
    apply_large_scale: bool = True

    def __post_init__(self) -> None:
        if self.profile not in {"NTN-CDL-A", "NTN-CDL-B", "NTN-CDL-C", "NTN-CDL-D"}:
            raise ValueError("profile must be NTN-CDL-A through NTN-CDL-D")
        _validate_standard(self.standard_version)
        _validate_deployment(self.deployment, self.environment)
        for name in (
            "carrier_frequency_hz", "bandwidth_hz", "delay_spread_s",
            "desired_asa_degrees", "desired_zsa_degrees",
        ):
            _finite(getattr(self, name), name, positive=True)
        for name in ("tx_array_profile", "rx_array_profile"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty resource name")
        _finite(self.faraday_rotation_rad, "faraday_rotation_rad")
        if self.k_factor_db is not None:
            _finite(self.k_factor_db, "k_factor_db")
            if self.profile not in {"NTN-CDL-C", "NTN-CDL-D"}:
                raise ValueError("k_factor_db is supported only for NTN-CDL-C/D")
        _boolean(self.apply_large_scale, "apply_large_scale")


@dataclass(frozen=True, slots=True)
class TR38811FlatFadingConfig:
    """Drop-based two-state land-mobile-satellite fading configuration."""

    standard_version: str
    deployment: str
    environment: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    good_state_probability: float
    good_direct_mean_db: float
    good_direct_std_db: float
    good_multipath_power_db: float
    bad_direct_mean_db: float
    bad_direct_std_db: float
    bad_multipath_power_db: float
    state_mode: str = "sample"
    num_rx_antennas: int = 1
    num_tx_antennas: int = 1
    num_sinusoids: int = 32
    terminal_speed_mps: float = 0.0
    apply_large_scale: bool = True

    def __post_init__(self) -> None:
        _validate_standard(self.standard_version)
        _validate_deployment(self.deployment, self.environment)
        if self.state_mode not in {"sample", "good", "bad"}:
            raise ValueError("state_mode must be 'sample', 'good', or 'bad'")
        for name in ("carrier_frequency_hz", "bandwidth_hz"):
            _finite(getattr(self, name), name, positive=True)
        _finite(self.good_state_probability, "good_state_probability")
        if not 0.0 <= self.good_state_probability <= 1.0:
            raise ValueError("good_state_probability must be between zero and one")
        for name in (
            "good_direct_mean_db", "good_direct_std_db", "good_multipath_power_db",
            "bad_direct_mean_db", "bad_direct_std_db", "bad_multipath_power_db",
        ):
            _finite(getattr(self, name), name)
        if self.good_direct_std_db < 0.0 or self.bad_direct_std_db < 0.0:
            raise ValueError("direct-path standard deviations must be nonnegative")
        for name in ("num_rx_antennas", "num_tx_antennas", "num_sinusoids"):
            _positive_integer(getattr(self, name), name)
        _finite(self.terminal_speed_mps, "terminal_speed_mps", nonnegative=True)
        _boolean(self.apply_large_scale, "apply_large_scale")


@dataclass(frozen=True, slots=True)
class TR38811SystemConfig:
    """Frequency-selective system-level NTN model configuration."""

    standard_version: str
    deployment: str
    environment: str
    carrier_frequency_hz: float
    bandwidth_hz: float
    tx_array_profile: str
    rx_array_profile: str
    delay_spread_s: float
    desired_asa_degrees: float
    desired_zsa_degrees: float
    spatial_consistency: bool = True
    apply_large_scale: bool = True

    def __post_init__(self) -> None:
        _validate_standard(self.standard_version)
        _validate_deployment(self.deployment, self.environment)
        _finite(self.carrier_frequency_hz, "carrier_frequency_hz", positive=True)
        _finite(self.bandwidth_hz, "bandwidth_hz", positive=True)
        for name in ("delay_spread_s", "desired_asa_degrees", "desired_zsa_degrees"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("tx_array_profile", "rx_array_profile"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty resource name")
        _boolean(self.spatial_consistency, "spatial_consistency")
        _boolean(self.apply_large_scale, "apply_large_scale")


@dataclass(frozen=True, slots=True)
class NTNAtmosphereConfig:
    """Caller-supplied atmospheric state for Earth-space attenuation."""

    temperature_k: float = 288.15
    pressure_hpa: float = 1013.25
    water_vapour_density_g_m3: float = 7.5
    rain_rate_mm_h: float | None = None
    cloud_liquid_water_kg_m2: float | None = None

    def __post_init__(self) -> None:
        for name in ("temperature_k", "pressure_hpa", "water_vapour_density_g_m3"):
            _finite(getattr(self, name), name, positive=True)
        for name in ("rain_rate_mm_h", "cloud_liquid_water_kg_m2"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name, nonnegative=True)


@dataclass(frozen=True, slots=True)
class NTNScintillationConfig:
    """Scintillation inputs, expressed as RMS attenuation in decibels."""

    mode: str = "auto"
    rms_db: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "ionospheric", "tropospheric"}:
            raise ValueError("invalid scintillation mode")
        _finite(self.rms_db, "rms_db", nonnegative=True)


@dataclass(frozen=True, slots=True)
class NTNAntennaConfig:
    """Circular-aperture NTN antenna parameters."""

    aperture_radius_wavelengths: float
    maximum_gain_dbi: float

    def __post_init__(self) -> None:
        _finite(self.aperture_radius_wavelengths, "aperture_radius_wavelengths", positive=True)
        _finite(self.maximum_gain_dbi, "maximum_gain_dbi")


def _parse(parameters: Mapping[str, Any], cls: type[Any], label: str) -> Any:
    expected = {field.name for field in fields(cls)}
    unknown = set(parameters) - expected
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required - set(parameters)
    if unknown or missing:
        raise ValueError(
            f"invalid {label} parameters; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return cls(**dict(parameters))


def parse_tdl_config(parameters: Mapping[str, Any]) -> TR38811TDLConfig:
    """Parse strict TR 38.811 TDL parameters."""

    return _parse(parameters, TR38811TDLConfig, "TR 38.811 TDL")


def parse_cdl_config(parameters: Mapping[str, Any]) -> TR38811CDLConfig:
    """Parse strict TR 38.811 CDL parameters."""

    return _parse(parameters, TR38811CDLConfig, "TR 38.811 CDL")


def parse_flat_config(parameters: Mapping[str, Any]) -> TR38811FlatFadingConfig:
    """Parse strict TR 38.811 flat-fading parameters."""

    return _parse(parameters, TR38811FlatFadingConfig, "TR 38.811 flat fading")


def parse_system_config(parameters: Mapping[str, Any]) -> TR38811SystemConfig:
    """Parse strict TR 38.811 system-model parameters."""

    return _parse(parameters, TR38811SystemConfig, "TR 38.811 system model")


__all__ = [
    "NTNAntennaConfig", "NTNAtmosphereConfig", "NTNScintillationConfig",
    "TR38811CDLConfig", "TR38811DeploymentConfig", "TR38811FlatFadingConfig",
    "TR38811SystemConfig", "TR38811TDLConfig", "parse_cdl_config",
    "parse_flat_config", "parse_system_config", "parse_tdl_config",
]
