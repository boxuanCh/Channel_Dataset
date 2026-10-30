"""Apply a fixed OFDM response to a transmit resource grid."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.channels.wireless.types import OFDMChannelResponse


def apply_ofdm(response: OFDMChannelResponse, signal: Any) -> Any:
    """Apply a fixed MIMO OFDM response to a transmit resource grid.

    Parameters
    ----------
    response : OFDMChannelResponse
        Channel coefficients with canonical OFDM response axes.
    signal : backend array
        Complex transmit grid with axes
        ``[batch, tx, tx_ant, symbol, subcarrier]``.

    Returns
    -------
    backend array
        Received grid with axes ``[batch, rx, rx_ant, symbol, subcarrier]``.

    Raises
    ------
    TypeError
        If the signal is not complex or its dtype differs from the response.
    ValueError
        If backend, device, rank, shape, or finite-value validation fails.

    Notes
    -----
    The operator sums over transmitter and transmit-antenna axes. It does not
    add noise and preserves PyTorch gradients.
    """

    backend_name = backend_name_for_array(signal)
    if backend_name != response.metadata.backend:
        raise ValueError("signal and OFDM response must use the same backend")
    backend = get_backend(backend_name)
    if backend.device(signal) != response.metadata.device:
        raise ValueError("signal and OFDM response must use the same device")
    if backend.ndim(signal) != 5:
        raise ValueError("OFDM signal must have rank 5")
    if not backend.is_complex(signal):
        raise TypeError("OFDM signal must have a complex dtype")
    if backend.dtype(signal) != backend.dtype(response.coefficients):
        raise TypeError("OFDM signal dtype must match response dtype")
    if not backend.all_finite(signal):
        raise ValueError("OFDM signal must contain only finite values")

    response_shape = response.shape
    signal_shape = backend.shape(signal)
    expected = (
        response_shape[0], response_shape[3], response_shape[4],
        response_shape[5], response_shape[6],
    )
    if signal_shape != expected:
        raise ValueError(f"OFDM signal shape {signal_shape} does not match expected {expected}")
    return backend.einsum("bratusk,btusk->brask", response.coefficients, signal)
