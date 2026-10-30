"""Apply a discrete-time channel response to a transmit waveform."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.channels.wireless.types import TimeChannelResponse


def apply_time(response: TimeChannelResponse, signal: Any) -> Any:
    r"""Apply a time-varying MIMO channel to a transmit waveform.

    Parameters
    ----------
    response : TimeChannelResponse
        Channel coefficients with canonical time-response axes.
    signal : backend array
        Complex waveform with axes ``[batch, tx, tx_ant, input_time]``. Its
        time length must equal the response output-time length.

    Returns
    -------
    backend array
        Received waveform with axes ``[batch, rx, rx_ant, output_time]``.

    Raises
    ------
    TypeError
        If the signal is not complex or its dtype differs from the response.
    ValueError
        If backend, device, rank, shape, or finite-value validation fails.

    Notes
    -----
    The operator computes :math:`y[n]=\sum_\ell h[n,\ell]x[n-\ell]` and uses
    zero-valued samples outside the supplied signal interval. It sums over
    transmitter and transmit-antenna axes, does not add noise, and preserves
    PyTorch gradients.
    """

    backend_name = backend_name_for_array(signal)
    if backend_name != response.metadata.backend:
        raise ValueError("signal and time response must use the same backend")
    backend = get_backend(backend_name)
    if backend.device(signal) != response.metadata.device:
        raise ValueError("signal and time response must use the same device")
    if backend.ndim(signal) != 4:
        raise ValueError("time-domain signal must have rank 4")
    if not backend.is_complex(signal):
        raise TypeError("time-domain signal must have a complex dtype")
    if backend.dtype(signal) != backend.dtype(response.coefficients):
        raise TypeError("time-domain signal dtype must match response dtype")
    if not backend.all_finite(signal):
        raise ValueError("time-domain signal must contain only finite values")

    response_shape = response.shape
    signal_shape = backend.shape(signal)
    expected = (response_shape[0], response_shape[3], response_shape[4], response_shape[5])
    if signal_shape != expected:
        raise ValueError(
            f"time-domain signal shape {signal_shape} does not match expected {expected}"
        )

    num_samples = signal_shape[-1]
    shifted = []
    for lag in range(response.lag_min, response.lag_max + 1):
        if abs(lag) >= num_samples:
            value = backend.zeros(
                signal_shape, dtype=backend.dtype(signal), device=response.metadata.device
            )
        elif lag > 0:
            padding = backend.zeros(
                (*signal_shape[:-1], lag),
                dtype=backend.dtype(signal),
                device=response.metadata.device,
            )
            value = backend.concatenate((padding, signal[..., : num_samples - lag]), axis=-1)
        elif lag < 0:
            padding = backend.zeros(
                (*signal_shape[:-1], -lag),
                dtype=backend.dtype(signal),
                device=response.metadata.device,
            )
            value = backend.concatenate((signal[..., -lag :], padding), axis=-1)
        else:
            value = signal
        shifted.append(value)
    delayed_signal = backend.stack(tuple(shifted), axis=-1)
    return backend.einsum("bratusl,btusl->bras", response.coefficients, delayed_signal)
