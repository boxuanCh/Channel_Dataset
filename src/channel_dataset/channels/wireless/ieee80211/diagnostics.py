"""Typed non-perturbing diagnostics for WLAN realizations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class WLANRealizationDiagnostics:
    """Deterministic parameters resolved for one WLAN realization request."""

    tap_delays_s: tuple[float, ...]
    tap_powers_linear: tuple[float, ...]
    receive_covariances: Any
    transmit_covariances: Any
    line_of_sight: bool
    k_factor_db: float | None
    maximum_doppler_hz: float
    review_status: str
    stream_labels: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_labels", MappingProxyType(dict(self.stream_labels)))


__all__ = ["WLANRealizationDiagnostics"]
