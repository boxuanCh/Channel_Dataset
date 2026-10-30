from __future__ import annotations

import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest, TopologyBatch
from channel_dataset.channels.wireless.ieee80211 import (
    TGahChannel,
    TGahConfig,
    WLANChannelSamplingRequest,
)
from channel_dataset.channels.wireless.ieee80211.large_scale import (
    sample_link_shadowing,
    tgah_path_loss_db,
)
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import RealizationMetadata


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _topology(execution, distance_m: float) -> TopologyBatch:
    backend = execution.array_backend
    tx = backend.array([[[0.0, 0.0, 0.0]]], dtype=execution.float_dtype)
    rx = backend.array([[[distance_m, 0.0, 0.0]]], dtype=execution.float_dtype)
    zeros = backend.zeros((1, 1, 3), dtype=execution.float_dtype)
    metadata = RealizationMetadata(
        "wireless", "test_topology", 1, execution.backend, execution.precision,
        execution.device, "topology", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx, rx, zeros, zeros, zeros, zeros,
        backend.array([[True]], dtype="bool"), backend.array([[0]], dtype="int64"),
        metadata, link_mask=backend.array([[[True]]], dtype="bool"),
    )


def _config(deployment: str, link_type: str = "ap_to_station") -> TGahConfig:
    return TGahConfig(
        "B", "11/0968r4", 900e6, 8e6, 1, 1, 1, 1, 0.0, 32, 64,
        0.5, 0.5, False, "path_loss", False, False, True, link_type, deployment,
    )


@pytest.mark.ieee80211
@pytest.mark.tgah
@pytest.mark.parametrize("deployment", ["indoor", "outdoor_macro", "outdoor_pico", "outdoor_hotspot"])
def test_tgah_deployments_generate_cir_and_ofdm(execution_context, deployment: str) -> None:
    sampling = ChannelSamplingRequest(
        1, 2, 1e3, execution_context, execution_context.random_stream(101)
    )
    cir = TGahChannel(_config(deployment)).realize(
        WLANChannelSamplingRequest(sampling, _topology(execution_context, 10.0))
    )
    assert cir.shape == (1, 1, 1, 1, 1, 9, 2)
    assert _as_numpy(cir.delays_s).reshape(-1)[1] == pytest.approx(5e-9)
    frequencies = subcarrier_frequencies(8, 31.25e3, execution_context)
    assert cir_to_ofdm(cir, frequencies).shape == (1, 1, 1, 1, 1, 2, 8)


@pytest.mark.ieee80211
@pytest.mark.tgah
def test_tgah_station_link_has_documented_additional_loss(backend) -> None:
    distance = backend.array([10.0], dtype="float64")
    ap = tgah_path_loss_db(distance, 900e6, "indoor", "ap_to_station", backend)
    station = tgah_path_loss_db(distance, 900e6, "indoor", "station_to_station", backend)
    assert float(_as_numpy(station - ap)[0]) == pytest.approx(3.0)


@pytest.mark.ieee80211
@pytest.mark.tgah
@pytest.mark.statistical
def test_tgah_shadowing_distribution(execution_context) -> None:
    values = sample_link_shadowing(
        execution_context.random_stream(777), (20000,), 5.0,
        dtype=execution_context.float_dtype,
    )
    samples = _as_numpy(values)
    assert float(samples.mean()) == pytest.approx(0.0, abs=0.12)
    assert float(samples.std()) == pytest.approx(5.0, rel=0.025)


@pytest.mark.ieee80211
@pytest.mark.tgah
def test_tgah_rejects_non_sub_ghz_configuration() -> None:
    with pytest.raises(ValueError, match="sub-1-GHz"):
        TGahConfig(
            "B", "11/0968r4", 2.4e9, 1e6, 1, 1, 1, 1, 0.0, 32, 64,
            0.5, 0.5, False, "none", False, False, True,
            "ap_to_station", "indoor",
        )
