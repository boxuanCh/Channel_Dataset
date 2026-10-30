"""Transforms for synchronized and absolute-delay NTN responses."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811.types import NTNChannelRealization
from channel_dataset.channels.wireless.transforms import cir_to_ofdm
from channel_dataset.channels.wireless.types import OFDMChannelResponse


def ntn_to_ofdm(
    realization: NTNChannelRealization,
    subcarrier_frequencies_hz: Any,
    *,
    include_propagation_delay: bool = False,
) -> OFDMChannelResponse:
    """Convert an NTN realization to a synchronized or absolute OFDM response."""

    response = cir_to_ofdm(realization.cir, subcarrier_frequencies_hz)
    if not include_propagation_delay:
        return response
    backend = get_backend(realization.cir.metadata.backend)
    batch, receivers, _, transmitters, _, time, _ = response.shape
    frequencies = backend.reshape(subcarrier_frequencies_hz, (1, 1, 1, 1, 1, 1, -1))
    delay = backend.reshape(
        realization.geometry.propagation_delay_s,
        (batch, receivers, 1, transmitters, 1, time, 1),
    )
    phase = backend.exp(-2j * pi * delay * frequencies)
    return OFDMChannelResponse(response.coefficients * phase, response.metadata)


__all__ = ["ntn_to_ofdm"]
