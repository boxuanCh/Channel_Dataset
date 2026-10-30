"""Convert a sampled CIR to an OFDM frequency response."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.channels.wireless.geometric_cdl.types import DynamicCIRBatch
from channel_dataset.channels.wireless.types import CIRBatch, OFDMChannelResponse


def cir_to_ofdm(
    cir: CIRBatch | DynamicCIRBatch, subcarrier_frequencies_hz: Any
) -> OFDMChannelResponse:
    r"""Convert a channel impulse response to an OFDM frequency response.

    Parameters
    ----------
    cir : CIRBatch
        Fixed path coefficients and delays.
    subcarrier_frequencies_hz : backend array
        Rank-one real frequency vector using the CIR backend, device, and
        precision.

    Returns
    -------
    OFDMChannelResponse
        Response with axes
        ``[batch, rx, rx_ant, tx, tx_ant, symbol, subcarrier]``.

    Raises
    ------
    TypeError
        If frequencies are not a real array of the required precision.
    ValueError
        If backend, device, rank, or finite-value validation fails.

    Notes
    -----
    The transform evaluates
    :math:`H(f,t)=\sum_p a_p(t)\exp(-j2\pi f\tau_p)`. Invalid masked paths
    contribute zero. The operation is deterministic and preserves PyTorch
    gradients through the coefficient array.
    """

    backend_name = backend_name_for_array(subcarrier_frequencies_hz)
    if backend_name != cir.metadata.backend:
        raise ValueError("subcarrier frequencies and CIR must use the same backend")
    backend = get_backend(backend_name)
    if backend.device(subcarrier_frequencies_hz) != cir.metadata.device:
        raise ValueError("subcarrier frequencies and CIR must use the same device")
    if backend.ndim(subcarrier_frequencies_hz) != 1:
        raise ValueError("subcarrier frequencies must have rank 1")
    if backend.shape(subcarrier_frequencies_hz)[0] == 0:
        raise ValueError("subcarrier frequencies must not be empty")
    if not backend.is_floating(subcarrier_frequencies_hz):
        raise TypeError("subcarrier frequencies must have a real floating dtype")
    expected_float = backend.resolve_dtype(
        "float32" if cir.metadata.precision == "single" else "float64"
    )
    if backend.dtype(subcarrier_frequencies_hz) != expected_float:
        raise TypeError("subcarrier frequency dtype does not match CIR precision")
    if not backend.all_finite(subcarrier_frequencies_hz):
        raise ValueError("subcarrier frequencies must be finite")

    b, r, ra, t, ta, p, time = cir.shape
    frequencies = backend.reshape(subcarrier_frequencies_hz, (1, 1, 1, 1, 1, 1, 1, -1))
    if isinstance(cir, DynamicCIRBatch):
        dynamic_delays = backend.transpose(cir.delays_s, (0, 1, 2, 4, 3))
        delays = backend.reshape(dynamic_delays, (b, r, 1, t, 1, time, p, 1))
    else:
        delays = backend.reshape(cir.delays_s, (b, r, 1, t, 1, 1, p, 1))
    coefficients = backend.transpose(cir.coefficients, (0, 1, 2, 3, 4, 6, 5))
    coefficients = backend.reshape(coefficients, (b, r, ra, t, ta, time, p, 1))
    phase = backend.exp((-2j * pi) * delays * frequencies)
    terms = coefficients * phase
    if cir.path_mask is not None:
        if isinstance(cir, DynamicCIRBatch):
            dynamic_mask = backend.transpose(cir.path_mask, (0, 1, 2, 4, 3))
            mask = backend.reshape(dynamic_mask, (b, r, 1, t, 1, time, p, 1))
        else:
            mask = backend.reshape(cir.path_mask, (b, r, 1, t, 1, 1, p, 1))
        terms = terms * mask
    response = backend.sum(terms, axis=-2)
    return OFDMChannelResponse(coefficients=response, metadata=cir.metadata)
