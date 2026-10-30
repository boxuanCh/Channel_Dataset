"""Backend-neutral random-distribution helpers."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import RandomStream


def complex_normal(
    stream: RandomStream[Any],
    shape: tuple[int, ...],
    *,
    variance: float = 1.0,
    dtype: Any = None,
) -> Any:
    """Draw circularly symmetric complex Gaussian samples.

    Parameters
    ----------
    stream : RandomStream
        Injected backend-native random stream.
    shape : tuple of int
        Output dimensions.
    variance : float, default=1
        Nonnegative complex variance :math:`E[|z|^2]`.
    dtype : optional
        Backend complex dtype. Supported values are ``complex64`` and
        ``complex128``.

    Returns
    -------
    backend array
        Complex samples on the stream's device.

    Notes
    -----
    Real and imaginary components are independent with variance
    ``variance / 2``.
    """

    return stream.complex_normal(shape, variance=variance, dtype=dtype)
