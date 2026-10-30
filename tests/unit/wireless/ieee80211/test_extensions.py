from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.ieee80211 import (
    register_ieee80211_models,
    tgac_delay_scale_factor,
    tgah_delay_scale_factor,
    transform_tgac_delays,
    transform_tgah_delays,
)
from channel_dataset.registry import ModelRegistry


@pytest.mark.ieee80211
@pytest.mark.tgac
def test_tgac_bandwidth_transform() -> None:
    delays = (0.0, 10e-9, 20e-9)
    assert [tgac_delay_scale_factor(x) for x in (20e6, 40e6, 80e6, 160e6)] == [1, 1, 2, 4]
    assert transform_tgac_delays(delays, 160e6) == pytest.approx((0.0, 2.5e-9, 5e-9))


@pytest.mark.ieee80211
@pytest.mark.tgah
def test_tgah_bandwidth_transform_is_independent_public_contract() -> None:
    delays = (0.0, 10e-9)
    assert [tgah_delay_scale_factor(x) for x in (1e6, 2e6, 4e6, 8e6, 16e6)] == [1, 1, 1, 2, 4]
    assert transform_tgah_delays(delays, 8e6) == pytest.approx((0.0, 5e-9))


@pytest.mark.ieee80211
def test_all_wlan_model_keys_register() -> None:
    registry = ModelRegistry()
    register_ieee80211_models(registry)
    assert registry.available_models() == (
        ("wireless", "ieee80211_tgac"),
        ("wireless", "ieee80211_tgah"),
        ("wireless", "ieee80211_tgax"),
        ("wireless", "ieee80211_tgn"),
    )
