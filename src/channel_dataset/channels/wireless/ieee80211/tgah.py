"""IEEE 802.11ah sub-1-GHz channel extensions."""

from __future__ import annotations

from channel_dataset.channels.wireless.ieee80211.config import TGahConfig, parse_tgah_config
from channel_dataset.channels.wireless.ieee80211.large_scale import tgah_path_loss_db
from channel_dataset.channels.wireless.ieee80211.profiles import (
    load_tgah_bandwidth_rule,
    load_tgah_path_loss_profile,
)
from channel_dataset.channels.wireless.ieee80211.tgn import _WLANCorrelatedTapChannel
from channel_dataset.registry import ModelRegistry


def tgah_delay_scale_factor(bandwidth_hz: float) -> int:
    """Return the TGah delay-profile reduction factor.

    This is intentionally distinct from the TGac public function. Under IEEE
    802.11-11/0968r4, 1/2/4 MHz retain the base spacing and 8/16 MHz use
    factors 2/4.
    """

    return load_tgah_bandwidth_rule().factor(bandwidth_hz)


def transform_tgah_delays(delays_s: tuple[float, ...], bandwidth_hz: float) -> tuple[float, ...]:
    """Apply the TGah physical delay-profile bandwidth transformation."""

    factor = tgah_delay_scale_factor(bandwidth_hz)
    return tuple(delay / factor for delay in delays_s)


class TGahChannel(_WLANCorrelatedTapChannel):
    """IEEE 802.11-11/0968r4 TGah SISO/MIMO channel model."""

    registry_name = "ieee80211_tgah"
    standard_version = "11/0968r4"

    def __init__(self, config: TGahConfig) -> None:
        if not isinstance(config, TGahConfig):
            raise TypeError("TGahChannel requires TGahConfig")
        super().__init__(config)

    def _delays_s(self) -> tuple[float, ...]:
        return transform_tgah_delays(self.profile.delays_s, self.config.bandwidth_hz)

    def _path_loss_db(self, request):
        backend = request.sampling.execution.array_backend
        return tgah_path_loss_db(
            self._link_distances(request), self.config.carrier_frequency_hz,
            self.config.deployment, self.config.link_type, backend,
        )

    def _shadow_standard_deviation_db(self) -> float:
        return load_tgah_path_loss_profile().shadow_std_db[self.config.deployment]


def _build_tgah(config: object) -> TGahChannel:
    if not isinstance(config, TGahConfig):
        raise TypeError("TGah builder requires TGahConfig")
    return TGahChannel(config)


def register_tgah_model(registry: ModelRegistry) -> None:
    """Register ``wireless/ieee80211_tgah`` with a model registry."""
    registry.register("wireless", "ieee80211_tgah", parse_tgah_config, _build_tgah)


__all__ = [
    "TGahChannel", "register_tgah_model", "tgah_delay_scale_factor",
    "transform_tgah_delays",
]
