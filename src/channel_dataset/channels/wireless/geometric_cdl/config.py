"""Typed configuration for geometrically consistent CDL evolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

_STANDARD_VERSIONS = {"16.1.0", "19.2.0"}
_PROFILES = {"CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"}


@dataclass(frozen=True, slots=True)
class GeometricCDLConfig:
    """Configuration for the dedicated geometric CDL channel model."""

    standard_version: str
    carrier_frequencies_hz: tuple[float, ...]
    initialization_mode: str = "scenario"
    fixed_profile: str | None = None
    mobility_mode: str = "hybrid"
    num_sinusoids: int = 300
    spatial_acf: str = "gaussian_exponential"
    automatic_segmentation: bool = True
    stationarity_distance_m: float | None = None
    transition_overlap_fraction: float = 0.5
    subpath_method: str = "legacy"
    rays_per_cluster: int = 20
    solver_tolerance: float = 1.0e-7
    solver_max_iterations: int = 24
    solver_fallback: str = "single_bounce"
    minimum_bounce_separation_m: float = 0.0
    ground_reflection: bool = False
    delay_reference: str = "excess"
    multiband_correlation: str = "shared_latent"
    apply_path_loss: bool | None = None
    apply_shadow_fading: bool | None = None
    diagnostics: str = "summary"
    trajectory_chunk_size: int = 256
    tx_array_profile: str = "profiles/wireless/antenna_isotropic_siso.yaml"
    rx_array_profile: str = "profiles/wireless/antenna_isotropic_siso.yaml"

    def __post_init__(self) -> None:
        if self.standard_version not in _STANDARD_VERSIONS:
            raise ValueError("standard_version must be '16.1.0' or '19.2.0'")
        if not isinstance(self.carrier_frequencies_hz, tuple) or not self.carrier_frequencies_hz:
            raise ValueError("carrier_frequencies_hz must be a nonempty tuple")
        if len(set(self.carrier_frequencies_hz)) != len(self.carrier_frequencies_hz):
            raise ValueError("carrier_frequencies_hz must be unique and ordered")
        for value in self.carrier_frequencies_hz:
            _positive(value, "carrier frequency")
        if self.initialization_mode not in {"scenario", "fixed_profile"}:
            raise ValueError("initialization_mode must be 'scenario' or 'fixed_profile'")
        if self.initialization_mode == "fixed_profile":
            if self.fixed_profile not in _PROFILES:
                raise ValueError("fixed_profile mode requires CDL-A through CDL-E")
        elif self.fixed_profile is not None:
            raise ValueError("fixed_profile is valid only in fixed_profile mode")
        if self.mobility_mode not in {"hybrid", "drift_only", "spatial_consistency_only"}:
            raise ValueError("unsupported mobility_mode")
        if self.spatial_acf not in {"gaussian_exponential", "exponential", "gaussian"}:
            raise ValueError("unsupported spatial_acf")
        if self.subpath_method not in {"legacy", "laplacian"}:
            raise ValueError("subpath_method must be 'legacy' or 'laplacian'")
        if self.solver_fallback not in {"single_bounce", "strict", "mask"}:
            raise ValueError("unsupported solver_fallback")
        if self.delay_reference != "excess":
            raise ValueError("the first release exposes excess delays")
        if self.multiband_correlation not in {"shared_latent", "independent"}:
            raise ValueError("unsupported multiband_correlation")
        if self.diagnostics not in {"none", "summary", "full"}:
            raise ValueError("diagnostics must be none, summary, or full")
        for name in ("num_sinusoids", "rays_per_cluster", "solver_max_iterations", "trajectory_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.solver_tolerance, "solver_tolerance")
        if self.stationarity_distance_m is not None:
            _positive(self.stationarity_distance_m, "stationarity_distance_m")
        if not 0.0 <= self.transition_overlap_fraction <= 1.0:
            raise ValueError("transition_overlap_fraction must lie in [0, 1]")
        if not isfinite(float(self.minimum_bounce_separation_m)) or self.minimum_bounce_separation_m < 0:
            raise ValueError("minimum_bounce_separation_m must be finite and nonnegative")
        for name in ("tx_array_profile", "rx_array_profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty resource reference")
        if self.apply_path_loss is None:
            object.__setattr__(self, "apply_path_loss", self.initialization_mode == "scenario")
        if self.apply_shadow_fading is None:
            object.__setattr__(self, "apply_shadow_fading", self.initialization_mode == "scenario")


def _positive(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def parse_geometric_cdl_config(parameters: Mapping[str, Any]) -> GeometricCDLConfig:
    """Strictly parse an external geometric-CDL parameter mapping."""

    if not isinstance(parameters, Mapping):
        raise TypeError("geometric CDL parameters must be a mapping")
    fields = set(GeometricCDLConfig.__dataclass_fields__)
    unknown = set(parameters) - fields
    if unknown:
        raise ValueError(f"unknown geometric CDL fields: {sorted(unknown)}")
    values = dict(parameters)
    if "carrier_frequencies_hz" in values:
        raw = values["carrier_frequencies_hz"]
        if not isinstance(raw, (list, tuple)):
            raise TypeError("carrier_frequencies_hz must be a sequence")
        values["carrier_frequencies_hz"] = tuple(float(value) for value in raw)
    return GeometricCDLConfig(**values)


__all__ = ["GeometricCDLConfig", "parse_geometric_cdl_config"]
