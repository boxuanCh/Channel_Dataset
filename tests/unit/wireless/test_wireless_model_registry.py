from channel_dataset.channels.wireless import register_wireless_models
from channel_dataset.registry import ModelRegistry


def test_register_wireless_models_exposes_complete_release_set() -> None:
    registry = ModelRegistry()

    register_wireless_models(registry)

    assert registry.available_models() == (
        ("wireless", "cdl"),
        ("wireless", "flat_mimo_fading"),
        ("wireless", "geometric_cdl"),
        ("wireless", "ieee80211_tgac"),
        ("wireless", "ieee80211_tgah"),
        ("wireless", "ieee80211_tgax"),
        ("wireless", "ieee80211_tgn"),
        ("wireless", "rayleigh_block_fading"),
        ("wireless", "tdl"),
        ("wireless", "tr37885_urban_cdl"),
        ("wireless", "tr37885_urban_system"),
        ("wireless", "tr38811_cdl"),
        ("wireless", "tr38811_flat"),
        ("wireless", "tr38811_system"),
        ("wireless", "tr38811_tdl"),
    )
