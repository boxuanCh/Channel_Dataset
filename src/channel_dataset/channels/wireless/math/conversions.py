"""Explicit conversions between logarithmic and linear quantities."""

from __future__ import annotations

from math import isfinite, log10
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend


def power_db_to_linear(power_db: Any) -> Any:
    """Convert a power ratio in decibels to a linear power ratio.

    Parameters
    ----------
    power_db : float or backend array
        Power ratio in dB.

    Returns
    -------
    float or backend array
        ``10 ** (power_db / 10)`` with input backend, dtype, and device
        preserved for array inputs.
    """

    return 10.0 ** (power_db / 10.0)


def linear_power_to_db(power_linear: Any) -> Any:
    """Convert a positive linear power ratio to decibels.

    Parameters
    ----------
    power_linear : float or backend array
        Strictly positive linear power ratio.

    Returns
    -------
    float or backend array
        ``10 * log10(power_linear)``.

    Raises
    ------
    ValueError
        If any input is non-finite or not strictly positive.
    """

    try:
        backend = get_backend(backend_name_for_array(power_linear))
    except TypeError:
        value = float(power_linear)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("power_linear must be finite and positive")
        return 10.0 * log10(value)
    if not backend.all_finite(power_linear) or backend.to_scalar(backend.min(power_linear)) <= 0.0:
        raise ValueError("power_linear must contain only finite positive values")
    return 10.0 * backend.log10(power_linear)


def amplitude_db_to_linear(amplitude_db: Any) -> Any:
    """Convert an amplitude ratio in decibels to a linear amplitude ratio.

    Parameters
    ----------
    amplitude_db : float or backend array
        Amplitude ratio in dB.

    Returns
    -------
    float or backend array
        ``10 ** (amplitude_db / 20)``.
    """

    return 10.0 ** (amplitude_db / 20.0)


def linear_amplitude_to_db(amplitude_linear: Any) -> Any:
    """Convert a nonzero linear amplitude magnitude to decibels.

    Parameters
    ----------
    amplitude_linear : float, complex, or backend array
        Real or complex amplitude. Its magnitude must be finite and nonzero.

    Returns
    -------
    float or backend array
        ``20 * log10(abs(amplitude_linear))``.

    Raises
    ------
    ValueError
        If any magnitude is non-finite or zero.
    """

    try:
        backend = get_backend(backend_name_for_array(amplitude_linear))
    except TypeError:
        magnitude = abs(amplitude_linear)
        if not isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("amplitude_linear magnitude must be finite and nonzero")
        return 20.0 * log10(magnitude)
    magnitude = backend.abs(amplitude_linear)
    if not backend.all_finite(magnitude) or backend.to_scalar(backend.min(magnitude)) <= 0.0:
        raise ValueError("amplitude_linear magnitudes must be finite and nonzero")
    return 20.0 * backend.log10(magnitude)


__all__ = [
    "amplitude_db_to_linear",
    "linear_amplitude_to_db",
    "linear_power_to_db",
    "power_db_to_linear",
]
