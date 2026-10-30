"""Typed configuration and standard-resource contracts for TDL models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

TDLProfileName = Literal[
    "TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E",
    "TDLA30", "TDLB100", "TDLC300", "TDLC60",
]
CorrelationMode = Literal["none", "kronecker", "joint"]

SCALABLE_PROFILES = frozenset({"TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"})
FIXED_PROFILES = frozenset({"TDLA30", "TDLB100", "TDLC300", "TDLC60"})
LOS_PROFILES = frozenset({"TDL-D", "TDL-E"})


@dataclass(frozen=True, slots=True)
class TDLProfile:
    """Validated tapped-delay-line parameters transcribed from a standard.

    Parameters
    ----------
    name : str
        Standard profile identifier.
    standard, standard_version, table : str
        Exact source specification and table identity.
    scalable : bool
        Whether delays are dimensionless and must be multiplied by the desired
        RMS delay spread.
    delays : tuple of float
        Normalized delays for scalable profiles or seconds for fixed profiles.
    powers_db : tuple of float
        Component powers in decibels before total-power normalization.
    component_types : tuple of {"rayleigh", "los"}
        Fading law for every represented path component.
    minimum_frequency_hz, maximum_frequency_hz : float
        Documented carrier-frequency applicability interval.
    maximum_bandwidth_hz : float, optional
        Documented maximum bandwidth when the source defines one.
    source_url, transcription_date, review_status : str
        Resource provenance and review state.
    """

    name: str
    standard: str
    standard_version: str
    table: str
    scalable: bool
    delays: tuple[float, ...]
    powers_db: tuple[float, ...]
    component_types: tuple[str, ...]
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    maximum_bandwidth_hz: float | None
    source_url: str
    transcription_date: str
    review_status: str

    def __post_init__(self) -> None:
        count = len(self.delays)
        if count == 0 or len(self.powers_db) != count or len(self.component_types) != count:
            raise ValueError("TDL resource delay, power, and component lengths must match")
        if any(not isfinite(value) or value < 0 for value in self.delays):
            raise ValueError("TDL resource delays must be finite and nonnegative")
        if any(not isfinite(value) for value in self.powers_db):
            raise ValueError("TDL resource powers must be finite")
        if any(value not in {"rayleigh", "los"} for value in self.component_types):
            raise ValueError("TDL resource component types must be 'rayleigh' or 'los'")
        if self.delays[0] != 0.0:
            raise ValueError("TDL resource must begin at zero delay")

    @property
    def num_paths(self) -> int:
        """Return the number of represented path components."""

        return len(self.delays)

    @property
    def has_los(self) -> bool:
        """Return whether the resource contains a specular LoS component."""

        return "los" in self.component_types


@dataclass(frozen=True, slots=True)
class TDLConfig:
    """Physical and numerical-size configuration for a 3GPP TDL model.

    Parameters
    ----------
    profile : str
        One of scalable ``TDL-A`` through ``TDL-E`` or fixed conformance
        profiles ``TDLA30``, ``TDLB100``, ``TDLC300``, and ``TDLC60``.
    standard_version : str
        Exact resource version, currently ``"16.1.0"``.
    delay_spread_s : float, optional
        Desired RMS delay spread for scalable profiles. Must be omitted for
        fixed-delay profiles.
    carrier_frequency_hz : float
        Carrier frequency within the selected profile's applicability range.
    bandwidth_hz : float
        Positive occupied bandwidth used for applicability validation.
    speed_mps : float
        Nonnegative relative speed used to calculate maximum Doppler shift.
    num_rx, num_rx_antennas, num_tx, num_tx_antennas : int
        Positive link and antenna dimensions.
    num_sinusoids : int
        Positive number of oscillators used for each classical Doppler process.
    k_factor_db : float, optional
        Desired aggregate Ricean K-factor for TDL-D/E. Other profiles reject it.
    correlation_mode : {"none", "kronecker", "joint"}
        Spatial-correlation construction.
    rx_correlation_coefficient, tx_correlation_coefficient : float, optional
        Exponential Kronecker correlation coefficients.
    joint_correlation_matrix : tuple of tuple of complex-compatible scalar, optional
        Full correlation matrix over flattened ``[rx_antenna, tx_antenna]``.
    """

    profile: str
    standard_version: str
    delay_spread_s: float | None
    carrier_frequency_hz: float
    bandwidth_hz: float
    speed_mps: float
    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    num_sinusoids: int
    k_factor_db: float | None
    correlation_mode: str
    rx_correlation_coefficient: float | None
    tx_correlation_coefficient: float | None
    joint_correlation_matrix: tuple[tuple[float, ...], ...] | None

    def __post_init__(self) -> None:
        if self.profile not in SCALABLE_PROFILES | FIXED_PROFILES:
            raise ValueError(f"unsupported TDL profile: {self.profile!r}")
        if self.standard_version != "16.1.0":
            raise ValueError("standard_version must be '16.1.0'")
        if self.profile in SCALABLE_PROFILES:
            _positive_finite(self.delay_spread_s, "delay_spread_s")
        elif self.delay_spread_s is not None:
            raise ValueError("delay_spread_s must be null for fixed-delay profiles")
        _positive_finite(self.carrier_frequency_hz, "carrier_frequency_hz")
        _positive_finite(self.bandwidth_hz, "bandwidth_hz")
        _nonnegative_finite(self.speed_mps, "speed_mps")
        for name in (
            "num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas", "num_sinusoids"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.k_factor_db is not None:
            _finite(self.k_factor_db, "k_factor_db")
            if self.profile not in LOS_PROFILES:
                raise ValueError("k_factor_db is supported only for TDL-D and TDL-E")
        if self.correlation_mode not in {"none", "kronecker", "joint"}:
            raise ValueError("correlation_mode must be 'none', 'kronecker', or 'joint'")
        self._validate_correlation()

    def _validate_correlation(self) -> None:
        rx = self.rx_correlation_coefficient
        tx = self.tx_correlation_coefficient
        joint = self.joint_correlation_matrix
        if self.correlation_mode == "none":
            if rx is not None or tx is not None or joint is not None:
                raise ValueError("correlation parameters must be null in 'none' mode")
            return
        if self.correlation_mode == "kronecker":
            if rx is None or tx is None or joint is not None:
                raise ValueError("kronecker mode requires rx/tx coefficients and no joint matrix")
            for value, name in ((rx, "rx_correlation_coefficient"), (tx, "tx_correlation_coefficient")):
                _finite(value, name)
                if abs(value) > 1.0:
                    raise ValueError(f"{name} magnitude must not exceed one")
            return
        if rx is not None or tx is not None or joint is None:
            raise ValueError("joint mode requires only joint_correlation_matrix")
        matrix = np.asarray(joint, dtype=np.float64)
        size = self.num_rx_antennas * self.num_tx_antennas
        if matrix.shape != (size, size):
            raise ValueError(f"joint_correlation_matrix must have shape ({size}, {size})")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("joint_correlation_matrix must be finite")
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("joint_correlation_matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-12):
            raise ValueError("joint_correlation_matrix diagonal must equal one")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12:
            raise ValueError("joint_correlation_matrix must be positive semidefinite")


def _finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _positive_finite(value: object, name: str) -> None:
    _finite(value, name)
    if float(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _nonnegative_finite(value: object, name: str) -> None:
    _finite(value, name)
    if float(value) < 0:
        raise ValueError(f"{name} must be nonnegative")
