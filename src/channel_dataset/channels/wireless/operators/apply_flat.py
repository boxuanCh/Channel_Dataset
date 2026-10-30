"""Apply a fixed flat MIMO response to a transmit signal."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.channels.wireless.types import FlatChannelResponse


def apply_flat(response: FlatChannelResponse, signal: Any) -> Any:
    """Apply a flat multi-link MIMO response.

    Parameters
    ----------
    response : FlatChannelResponse
        Coefficients with axes ``[batch, rx, rx_ant, tx, tx_ant]``.
    signal : backend array
        Complex input with leading axes ``[batch, tx, tx_ant]`` and zero or
        more trailing sample dimensions.

    Returns
    -------
    backend array
        Output with axes ``[batch, rx, rx_ant, ...]``.

    Notes
    -----
    Transmitter and transmit-antenna axes are summed. Noise is deliberately
    excluded, and PyTorch gradients are preserved.
    """

    backend_name = backend_name_for_array(signal)
    if backend_name != response.metadata.backend:
        raise ValueError("signal and flat response must use the same backend")
    backend = get_backend(backend_name)
    if backend.device(signal) != response.metadata.device:
        raise ValueError("signal and flat response must use the same device")
    if backend.ndim(signal) < 3:
        raise ValueError("flat-channel signal must have rank at least 3")
    if not backend.is_complex(signal):
        raise TypeError("flat-channel signal must have a complex dtype")
    if backend.dtype(signal) != backend.dtype(response.coefficients):
        raise TypeError("flat-channel signal dtype must match response dtype")
    if not backend.all_finite(signal):
        raise ValueError("flat-channel signal must contain only finite values")
    response_shape = response.shape
    signal_shape = backend.shape(signal)
    expected = (response_shape[0], response_shape[3], response_shape[4])
    if signal_shape[:3] != expected:
        raise ValueError(
            f"flat-channel signal leading shape {signal_shape[:3]} does not match {expected}"
        )
    return backend.einsum("britu,btu...->bri...", response.coefficients, signal)


__all__ = ["apply_flat"]
