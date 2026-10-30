"""Fixed urban CDL models from TR 37.885 clause 6.2.3.1."""

from __future__ import annotations

from channel_dataset.channels.wireless.tr37885._channel import UrbanChannelAssembler
from channel_dataset.channels.wireless.tr37885.config import (
    TR37885UrbanCDLConfig,
    parse_urban_cdl_config,
)
from channel_dataset.channels.wireless.tr37885.types import (
    V2XChannelRealization,
    V2XChannelSamplingRequest,
)
from channel_dataset.core import ChannelModel
from channel_dataset.registry import ModelRegistry


class TR37885UrbanCDLChannel(ChannelModel[V2XChannelSamplingRequest, V2XChannelRealization]):
    """Sample the state-selected urban LOS/NLOSv/NLOS fixed CDL."""

    def __init__(self, config: TR37885UrbanCDLConfig) -> None:
        if not isinstance(config, TR37885UrbanCDLConfig):
            raise TypeError("TR37885UrbanCDLChannel requires TR37885UrbanCDLConfig")
        self.config = config
        self._assembler = UrbanChannelAssembler(config, self.name, fixed_cdl=True)

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr37885_urban_cdl"

    def realize(self, request: V2XChannelSamplingRequest) -> V2XChannelRealization:
        """Sample one fixed-window V2X CDL realization."""

        return self._assembler.realize(request)

    def realize_with_diagnostics(self, request: V2XChannelSamplingRequest) -> V2XChannelRealization:
        """Sample a realization with Doppler and condition diagnostics."""

        return self._assembler.realize(request, diagnostics=True)


def _build(config: object) -> TR37885UrbanCDLChannel:
    if not isinstance(config, TR37885UrbanCDLConfig):
        raise TypeError("TR 37.885 CDL builder requires TR37885UrbanCDLConfig")
    return TR37885UrbanCDLChannel(config)


def register_urban_cdl_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr37885_urban_cdl``."""

    registry.register("wireless", "tr37885_urban_cdl", parse_urban_cdl_config, _build)


__all__ = ["TR37885UrbanCDLChannel", "register_urban_cdl_model"]
