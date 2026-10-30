"""Explicit power normalization for channel response containers."""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeVar, cast

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.types import (
    FlatChannelResponse,
    OFDMChannelResponse,
    TimeChannelResponse,
)

ResponseT = TypeVar(
    "ResponseT", FlatChannelResponse, OFDMChannelResponse, TimeChannelResponse
)


def normalize_response(response: ResponseT, *, target_power: float = 1.0) -> ResponseT:
    r"""Normalize every batch/link response to a target mean power.

    Parameters
    ----------
    response : FlatChannelResponse, TimeChannelResponse, or OFDMChannelResponse
        Validated response to normalize.
    target_power : float, default=1.0
        Positive, finite target for the mean squared coefficient magnitude.

    Returns
    -------
    response
        A new response of the same concrete type and with the same metadata.
        Each ``[batch, rx, tx]`` link is scaled independently.

    Raises
    ------
    TypeError
        If ``response`` is unsupported or ``target_power`` is not real.
    ValueError
        If the target is invalid or any link has zero power.

    Notes
    -----
    The averaging axes are receive antenna, transmit antenna, and any temporal,
    lag, symbol, or subcarrier axes. The transform is deterministic, explicit,
    and differentiable for nonzero PyTorch responses.
    """

    if not isinstance(response, (FlatChannelResponse, TimeChannelResponse, OFDMChannelResponse)):
        raise TypeError("response must be a flat, time, or OFDM channel response")
    if isinstance(target_power, bool) or not isinstance(target_power, Real):
        raise TypeError("target_power must be a real number")
    if not isfinite(float(target_power)) or target_power <= 0:
        raise ValueError("target_power must be finite and positive")

    backend = get_backend(response.metadata.backend)
    values = response.coefficients
    if isinstance(response, FlatChannelResponse):
        reduction_axes = (2, 4)
        scale_shape = (response.shape[0], response.shape[1], 1, response.shape[3], 1)
    else:
        reduction_axes = (2, 4, 5, 6)
        scale_shape = (
            response.shape[0], response.shape[1], 1, response.shape[3], 1, 1, 1
        )
    power = backend.mean(backend.abs(values) ** 2, axis=reduction_axes)
    if float(backend.to_scalar(backend.min(power))) <= 0.0:
        raise ValueError("cannot normalize a response containing a zero-power link")
    scale = backend.sqrt(float(target_power) / power)
    normalized = values * backend.reshape(scale, scale_shape)
    if isinstance(response, TimeChannelResponse):
        return cast(ResponseT, TimeChannelResponse(normalized, response.metadata, response.lag_min))
    return cast(ResponseT, type(response)(normalized, response.metadata))
