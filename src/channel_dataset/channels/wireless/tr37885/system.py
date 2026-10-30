"""Stochastic system-level urban V2V channel from TR 37.885."""

from __future__ import annotations

from channel_dataset.channels.wireless.tr37885._channel import UrbanChannelAssembler
from channel_dataset.channels.wireless.tr37885.config import (
    TR37885UrbanSystemConfig,
    parse_urban_system_config,
)
from channel_dataset.channels.wireless.tr37885.types import (
    V2XChannelRealization,
    V2XChannelSamplingRequest,
)
from channel_dataset.core import ChannelModel
from channel_dataset.registry import ModelRegistry


class TR37885UrbanSystemChannel(
    ChannelModel[V2XChannelSamplingRequest, V2XChannelRealization]
):
    """Sample stochastic three-state urban V2V realizations."""

    def __init__(self, config: TR37885UrbanSystemConfig) -> None:
        if not isinstance(config, TR37885UrbanSystemConfig):
            raise TypeError("TR37885UrbanSystemChannel requires TR37885UrbanSystemConfig")
        self.config = config
        self._assembler = UrbanChannelAssembler(config, self.name, fixed_cdl=False)

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr37885_urban_system"

    def realize(self, request: V2XChannelSamplingRequest) -> V2XChannelRealization:
        """Sample one fixed-window stochastic V2X realization."""

        return self._assembler.realize(request)

    def realize_with_diagnostics(self, request: V2XChannelSamplingRequest) -> V2XChannelRealization:
        """Sample a realization with Doppler and condition diagnostics."""

        return self._assembler.realize(request, diagnostics=True)


def _build(config: object) -> TR37885UrbanSystemChannel:
    if not isinstance(config, TR37885UrbanSystemConfig):
        raise TypeError("TR 37.885 system builder requires TR37885UrbanSystemConfig")
    return TR37885UrbanSystemChannel(config)


def register_urban_system_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr37885_urban_system``."""

    registry.register("wireless", "tr37885_urban_system", parse_urban_system_config, _build)


__all__ = ["TR37885UrbanSystemChannel", "register_urban_system_model"]
