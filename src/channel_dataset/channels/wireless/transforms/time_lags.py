"""Discrete-time channel lag construction."""

from __future__ import annotations

from math import ceil, isfinite
from numbers import Real
from typing import Any

from channel_dataset.core import ExecutionContext


def time_lag_bounds(
    max_delay_s: float,
    sampling_frequency_hz: float,
    *,
    sinc_half_width: int = 6,
) -> tuple[int, int]:
    """Calculate inclusive sample-lag bounds for a delayed CIR.

    Parameters
    ----------
    max_delay_s : float
        Largest physical path delay in seconds. Must be finite and
        nonnegative.
    sampling_frequency_hz : float
        Positive, finite time-domain sampling frequency in hertz.
    sinc_half_width : int, default=6
        Nonnegative number of extra samples retained on either side of the
        physical delay interval for the truncated sinc representation.

    Returns
    -------
    lag_min, lag_max : tuple of int
        Inclusive integer bounds. They are ``-sinc_half_width`` and
        ``ceil(max_delay_s * sampling_frequency_hz) + sinc_half_width``.

    Raises
    ------
    TypeError
        If an argument has an incompatible Python type.
    ValueError
        If a numeric argument is non-finite or outside its valid range.
    """

    if isinstance(max_delay_s, bool) or not isinstance(max_delay_s, Real):
        raise TypeError("max_delay_s must be a real number")
    if isinstance(sampling_frequency_hz, bool) or not isinstance(
        sampling_frequency_hz, Real
    ):
        raise TypeError("sampling_frequency_hz must be a real number")
    if isinstance(sinc_half_width, bool) or not isinstance(sinc_half_width, int):
        raise TypeError("sinc_half_width must be an integer")
    if not isfinite(float(max_delay_s)) or max_delay_s < 0:
        raise ValueError("max_delay_s must be finite and nonnegative")
    if not isfinite(float(sampling_frequency_hz)) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if sinc_half_width < 0:
        raise ValueError("sinc_half_width must be nonnegative")
    return (
        -sinc_half_width,
        ceil(float(max_delay_s) * float(sampling_frequency_hz)) + sinc_half_width,
    )


def time_lags(
    lag_min: int,
    lag_max: int,
    execution: ExecutionContext,
) -> Any:
    """Construct an inclusive vector of integer sample lags.

    Parameters
    ----------
    lag_min, lag_max : int
        Inclusive lower and upper lag bounds. ``lag_max`` must not be less
        than ``lag_min``.
    execution : ExecutionContext
        Selects the native output backend and device.

    Returns
    -------
    backend array
        Rank-one ``int64`` vector on the requested backend and device.

    Raises
    ------
    TypeError
        If either bound is not an integer.
    ValueError
        If the bounds describe an empty interval.
    """

    if isinstance(lag_min, bool) or not isinstance(lag_min, int):
        raise TypeError("lag_min must be an integer")
    if isinstance(lag_max, bool) or not isinstance(lag_max, int):
        raise TypeError("lag_max must be an integer")
    if lag_max < lag_min:
        raise ValueError("lag_max must be greater than or equal to lag_min")
    values = execution.array_backend.arange(
        lag_max - lag_min + 1,
        dtype="int64",
        device=execution.device,
    )
    return values + lag_min
