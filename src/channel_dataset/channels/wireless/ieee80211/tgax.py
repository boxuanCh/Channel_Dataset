"""IEEE 802.11ax indoor and explicitly claimed outdoor subset."""

from __future__ import annotations

from channel_dataset.channels.wireless.ieee80211.config import TGaxConfig, parse_tgax_config
from channel_dataset.channels.wireless.ieee80211.large_scale import tgax_path_loss_db
from channel_dataset.channels.wireless.ieee80211.profiles import (
    load_tgax_indoor_profile,
    load_tgax_outdoor_profile,
)
from channel_dataset.channels.wireless.ieee80211.tgac import transform_tgac_delays
from channel_dataset.channels.wireless.ieee80211.tgn import _WLANCorrelatedTapChannel
from channel_dataset.registry import ModelRegistry


class TGaxChannel(_WLANCorrelatedTapChannel):
    """IEEE 802.11-14/0882r4 TGax indoor/outdoor channel subset.

    Indoor scenarios reuse the accepted TGn/TGac small-scale kernel. Outdoor
    UMi/UMa support is deliberately IEEE-owned and does not alias the package's
    TR 38.901 model identity or resources.
    """

    registry_name = "ieee80211_tgax"
    standard_version = "14/0882r4"

    def __init__(self, config: TGaxConfig) -> None:
        if not isinstance(config, TGaxConfig):
            raise TypeError("TGaxChannel requires TGaxConfig")
        super().__init__(config)

    def _delays_s(self) -> tuple[float, ...]:
        return transform_tgac_delays(self.profile.delays_s, self.config.bandwidth_hz)

    def _requires_topology(self) -> bool:
        return True

    def _path_loss_db(self, request):
        backend = request.sampling.execution.array_backend
        tx_height = request.topology.tx_positions_m[:, None, :, 2]
        rx_height = request.topology.rx_positions_m[:, :, None, 2]
        return tgax_path_loss_db(
            self._link_distances(request), self.config.carrier_frequency_hz,
            self.config.scenario, self.config.line_of_sight, backend,
            ap_height_m=tx_height,
            station_height_m=rx_height,
            street_width_m=self.config.street_width_m,
            average_building_height_m=self.config.average_building_height_m,
        )

    def _penetration_loss_db(self, request) -> float:
        if self.config.scenario.startswith("outdoor"):
            return 0.0
        profile = load_tgax_indoor_profile()
        walls = profile.wall_loss_db[self.config.scenario] * self.config.wall_count
        floors = 0.0
        if self.config.floor_count:
            count = self.config.floor_count
            exponent = (count + 2.0) / (count + 1.0) - profile.floor_loss_exponent_offset
            floors = profile.floor_loss_coefficient_db * count**exponent
        return walls + floors

    def _shadow_standard_deviation_db(self) -> float:
        if self.config.scenario.startswith("indoor"):
            return super()._shadow_standard_deviation_db()
        scenario = load_tgax_outdoor_profile().scenarios[self.config.scenario]
        field = "shadow_std_los_db" if self.config.line_of_sight else "shadow_std_nlos_db"
        return float(scenario[field])

    def _tap_environmental_speed_mps(self, tap: int) -> float:
        if not self.config.fast_object_effect or self.config.scenario.startswith("indoor"):
            return super()._tap_environmental_speed_mps(tap)
        cluster_ids = self.profile.taps[tap].cluster_ids
        return 60.0 / 3.6 if any(cluster in {2, 3} for cluster in cluster_ids) else 3.0 / 3.6


def _build_tgax(config: object) -> TGaxChannel:
    if not isinstance(config, TGaxConfig):
        raise TypeError("TGax builder requires TGaxConfig")
    return TGaxChannel(config)


def register_tgax_model(registry: ModelRegistry) -> None:
    """Register ``wireless/ieee80211_tgax`` with a model registry."""
    registry.register("wireless", "ieee80211_tgax", parse_tgax_config, _build_tgax)


__all__ = ["TGaxChannel", "register_tgax_model"]
