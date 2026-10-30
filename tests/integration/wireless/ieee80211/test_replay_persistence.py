from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest, load_cir, save_cir
from channel_dataset.channels.wireless.ieee80211 import (
    TGnChannel,
    TGnConfig,
    WLANChannelSamplingRequest,
)
from channel_dataset.core import ExecutionContext


def _config() -> TGnConfig:
    return TGnConfig(
        "B", "03/940r4", 5.25e9, 20e6, 1, 1, 1, 1, 0.0, 32, 64,
        0.5, 0.5, False, "none", False, False, True,
    )


@pytest.mark.ieee80211
@pytest.mark.tgn
@pytest.mark.torch
def test_wlan_cir_persists_and_cross_loads(tmp_path) -> None:
    pytest.importorskip("torch")
    numpy_execution = ExecutionContext("numpy", "double", "cpu")
    sampling = ChannelSamplingRequest(
        2, 3, 1e3, numpy_execution, numpy_execution.random_stream(616)
    )
    original = TGnChannel(_config()).realize(WLANChannelSamplingRequest(sampling))
    save_cir(tmp_path / "wlan-cir", original, chunk_size=1)
    replay = load_cir(tmp_path / "wlan-cir", ExecutionContext("torch", "double", "cpu"))
    assert replay.metadata.model_name == "ieee80211_tgn"
    assert replay.metadata.standard_version == "03/940r4"
    assert np.array_equal(original.coefficients, replay.coefficients.cpu().numpy())
    assert np.array_equal(original.delays_s, replay.delays_s.cpu().numpy())
