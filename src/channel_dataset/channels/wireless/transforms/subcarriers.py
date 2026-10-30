"""OFDM subcarrier-frequency construction."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

from channel_dataset.core import ExecutionContext


def subcarrier_frequencies(
    num_subcarriers: int,
    subcarrier_spacing_hz: float,
    execution: ExecutionContext,
) -> Any:
    """Construct centered OFDM subcarrier frequencies.

    Parameters
    ----------
    num_subcarriers : int
        Positive FFT/subcarrier count.
    subcarrier_spacing_hz : float
        Positive spacing between adjacent subcarriers in hertz.
    execution : ExecutionContext
        Selects output backend, precision, and device.

    Returns
    -------
    backend array
        Real vector with shape ``[num_subcarriers]``.

    Raises
    ------
    TypeError
        If ``num_subcarriers`` is not an integer.
    ValueError
        If a count or spacing is not positive.

    Notes
    -----
    Even sizes use indices ``[-N/2, ..., N/2-1]``; odd sizes use the
    symmetric range ``[-floor(N/2), ..., floor(N/2)]``.
    """

    if isinstance(num_subcarriers, bool) or not isinstance(num_subcarriers, int):
        raise TypeError("num_subcarriers must be an integer")
    if num_subcarriers <= 0:
        raise ValueError("num_subcarriers must be a positive integer")
    if isinstance(subcarrier_spacing_hz, bool) or not isinstance(subcarrier_spacing_hz, Real):
        raise TypeError("subcarrier_spacing_hz must be a real number")
    if not isfinite(float(subcarrier_spacing_hz)) or subcarrier_spacing_hz <= 0:
        raise ValueError("subcarrier_spacing_hz must be finite and positive")
    backend = execution.array_backend
    indices = backend.arange(
        num_subcarriers, dtype=execution.float_dtype, device=execution.device
    )
    return (indices - num_subcarriers // 2) * subcarrier_spacing_hz
