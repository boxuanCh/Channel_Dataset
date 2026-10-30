"""Sampling requests for IEEE 802.11 channel models."""

from __future__ import annotations

from dataclasses import dataclass

from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import TopologyBatch


@dataclass(frozen=True, slots=True)
class WLANChannelSamplingRequest:
    """Compose numerical sampling controls with optional WLAN topology.

    Parameters
    ----------
    sampling : ChannelSamplingRequest
        Batch size, time grid, execution context, and injected random stream.
    topology : TopologyBatch, optional
        AP/station geometry. When supplied, its batch size, backend, device,
        and precision must match ``sampling``.
    """

    sampling: ChannelSamplingRequest
    topology: TopologyBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, ChannelSamplingRequest):
            raise TypeError("sampling must be a ChannelSamplingRequest")
        if self.topology is None:
            return
        if not isinstance(self.topology, TopologyBatch):
            raise TypeError("topology must be a TopologyBatch when provided")
        metadata = self.topology.metadata
        execution = self.sampling.execution
        if self.topology.shape[0] != self.sampling.batch_size:
            raise ValueError("topology and sampling batch sizes must match")
        if metadata.backend != execution.backend:
            raise ValueError("topology and sampling backends must match")
        if metadata.device != execution.device:
            raise ValueError("topology and sampling devices must match")
        if metadata.precision != execution.precision:
            raise ValueError("topology and sampling precisions must match")


__all__ = ["WLANChannelSamplingRequest"]
