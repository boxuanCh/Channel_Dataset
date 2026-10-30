"""IEEE 802.11ac bandwidth and multiuser extensions."""

from __future__ import annotations

from hashlib import sha256
from struct import pack

from channel_dataset.channels.wireless.ieee80211.config import TGacConfig, parse_tgac_config
from channel_dataset.channels.wireless.ieee80211.profiles import load_tgac_bandwidth_rule
from channel_dataset.channels.wireless.ieee80211.requests import WLANChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211.tgn import _WLANCorrelatedTapChannel
from channel_dataset.registry import ModelRegistry


def tgac_delay_scale_factor(bandwidth_hz: float) -> int:
    """Return the TGac PDP tap-spacing reduction factor.

    IEEE 802.11-09/0308r12 section 2 specifies
    ``2**ceil(log2(W/40 MHz))`` with a minimum factor of one. The supported
    20/40/80/160-MHz values therefore map to 1/1/2/4.
    """

    return load_tgac_bandwidth_rule().factor(bandwidth_hz)


def transform_tgac_delays(delays_s: tuple[float, ...], bandwidth_hz: float) -> tuple[float, ...]:
    """Apply the TGac physical delay-profile bandwidth transformation."""

    factor = tgac_delay_scale_factor(bandwidth_hz)
    return tuple(delay / factor for delay in delays_s)


class TGacChannel(_WLANCorrelatedTapChannel):
    """IEEE 802.11-09/0308r12 TGac SU/MU-MIMO channel model."""

    registry_name = "ieee80211_tgac"
    standard_version = "09/0308r12"

    def __init__(self, config: TGacConfig) -> None:
        if not isinstance(config, TGacConfig):
            raise TypeError("TGacChannel requires TGacConfig")
        super().__init__(config)

    def _delays_s(self) -> tuple[float, ...]:
        return transform_tgac_delays(self.profile.delays_s, self.config.bandwidth_hz)

    def _requires_topology(self) -> bool:
        return super()._requires_topology() or self.config.num_users > 1

    def _validate_request(self, request: WLANChannelSamplingRequest) -> None:
        super()._validate_request(request)
        if self.config.num_users > 1 and request.topology is None:
            raise ValueError("TGac MU-MIMO requires topology")

    def _link_stream_key(
        self, request: WLANChannelSamplingRequest, batch: int, rx: int, tx: int
    ) -> str:
        if self.config.num_users == 1 or request.topology is None:
            return super()._link_stream_key(request, batch, rx, tx)
        backend = request.sampling.execution.array_backend
        position = (
            request.topology.rx_positions_m[batch, rx]
            if self.config.direction == "downlink"
            else request.topology.tx_positions_m[batch, tx]
        )
        coordinates = tuple(float(backend.to_scalar(position[index])) for index in range(3))
        digest = sha256(pack(">ddd", *coordinates)).hexdigest()[:16]
        peer = tx if self.config.direction == "downlink" else rx
        return f"{batch}.user.{digest}.peer.{peer}"


def _build_tgac(config: object) -> TGacChannel:
    if not isinstance(config, TGacConfig):
        raise TypeError("TGac builder requires TGacConfig")
    return TGacChannel(config)


def register_tgac_model(registry: ModelRegistry) -> None:
    """Register ``wireless/ieee80211_tgac`` with a model registry."""
    registry.register("wireless", "ieee80211_tgac", parse_tgac_config, _build_tgac)


__all__ = [
    "TGacChannel", "register_tgac_model", "tgac_delay_scale_factor",
    "transform_tgac_delays",
]
