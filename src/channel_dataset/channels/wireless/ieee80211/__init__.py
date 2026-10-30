"""IEEE 802.11 TGn, TGac, TGah, and TGax channel models."""

from channel_dataset.channels.wireless.ieee80211.config import (
    TGacConfig,
    TGahConfig,
    TGaxConfig,
    TGnConfig,
    parse_tgac_config,
    parse_tgah_config,
    parse_tgax_config,
    parse_tgn_config,
)
from channel_dataset.channels.wireless.ieee80211.diagnostics import WLANRealizationDiagnostics
from channel_dataset.channels.wireless.ieee80211.profiles import (
    IEEE80211ClusterProfile,
    IEEE80211Profile,
    IEEE80211TapProfile,
    TGacBandwidthRule,
    TGacMultiUserRule,
    TGahBandwidthRule,
    TGahPathLossProfile,
    TGaxIndoorProfile,
    TGaxOutdoorProfile,
    load_ieee80211_profile,
    load_tgac_bandwidth_rule,
    load_tgac_multiuser_rule,
    load_tgah_bandwidth_rule,
    load_tgah_path_loss_profile,
    load_tgax_indoor_profile,
    load_tgax_outdoor_profile,
)
from channel_dataset.channels.wireless.ieee80211.requests import WLANChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211.tgac import (
    TGacChannel,
    register_tgac_model,
    tgac_delay_scale_factor,
    transform_tgac_delays,
)
from channel_dataset.channels.wireless.ieee80211.tgah import (
    TGahChannel,
    register_tgah_model,
    tgah_delay_scale_factor,
    transform_tgah_delays,
)
from channel_dataset.channels.wireless.ieee80211.tgax import TGaxChannel, register_tgax_model
from channel_dataset.channels.wireless.ieee80211.tgn import TGnChannel, register_tgn_model


def register_ieee80211_models(registry) -> None:
    """Register all four IEEE 802.11 model keys in ``registry``."""

    register_tgn_model(registry)
    register_tgac_model(registry)
    register_tgah_model(registry)
    register_tgax_model(registry)


__all__ = [
    "IEEE80211ClusterProfile", "IEEE80211Profile", "IEEE80211TapProfile",
    "TGacBandwidthRule", "TGacChannel", "TGacConfig", "TGacMultiUserRule",
    "TGahBandwidthRule", "TGahChannel", "TGahConfig", "TGahPathLossProfile",
    "TGaxChannel", "TGaxConfig", "TGaxIndoorProfile", "TGaxOutdoorProfile",
    "TGnChannel", "TGnConfig", "WLANChannelSamplingRequest",
    "WLANRealizationDiagnostics", "load_ieee80211_profile", "load_tgac_bandwidth_rule",
    "load_tgac_multiuser_rule", "load_tgah_bandwidth_rule", "load_tgah_path_loss_profile",
    "load_tgax_indoor_profile", "load_tgax_outdoor_profile", "parse_tgac_config",
    "parse_tgah_config", "parse_tgax_config", "parse_tgn_config",
    "register_ieee80211_models", "register_tgac_model", "register_tgah_model",
    "register_tgax_model", "register_tgn_model", "tgac_delay_scale_factor",
    "tgah_delay_scale_factor", "transform_tgac_delays", "transform_tgah_delays",
]
