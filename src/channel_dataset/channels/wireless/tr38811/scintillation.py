"""Ionospheric and tropospheric scintillation helpers."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import RandomStream
from channel_dataset.core import ExecutionContext


def scale_ionospheric_s4(
    reference_s4: float,
    reference_frequency_hz: float,
    carrier_frequency_hz: float,
) -> float:
    """Scale a weak-scintillation S4 index with the clause-6.6.6 exponent."""

    if reference_s4 < 0.0:
        raise ValueError("reference_s4 must be nonnegative")
    if reference_frequency_hz <= 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("frequencies must be positive")
    return reference_s4 * (reference_frequency_hz / carrier_frequency_hz) ** 1.5


def scale_ionospheric_phase_std(
    reference_phase_std_rad: float,
    reference_frequency_hz: float,
    carrier_frequency_hz: float,
) -> float:
    """Scale ionospheric phase deviation inversely with frequency."""

    if reference_phase_std_rad < 0.0:
        raise ValueError("reference_phase_std_rad must be nonnegative")
    if reference_frequency_hz <= 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("frequencies must be positive")
    return reference_phase_std_rad * reference_frequency_hz / carrier_frequency_hz


def sample_scintillation_db(
    shape: tuple[int, ...],
    rms_db: float,
    random_stream: RandomStream[Any],
    execution: ExecutionContext,
) -> Any:
    """Sample a zero-mean Gaussian scintillation term in decibels."""

    if rms_db < 0.0:
        raise ValueError("rms_db must be nonnegative")
    return random_stream.fork("wireless.tr38811.scintillation").normal(
        shape, stddev=rms_db, dtype=execution.float_dtype
    )


__all__ = [
    "sample_scintillation_db", "scale_ionospheric_phase_std", "scale_ionospheric_s4",
]
