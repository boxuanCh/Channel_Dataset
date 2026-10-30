from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest, TopologyBatch
from channel_dataset.channels.wireless.ieee80211 import (
    TGacChannel,
    TGacConfig,
    TGahChannel,
    TGahConfig,
    TGaxChannel,
    TGaxConfig,
    TGnChannel,
    TGnConfig,
    WLANChannelSamplingRequest,
    load_ieee80211_profile,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import as_numpy

pytestmark = [pytest.mark.statistical, pytest.mark.ieee80211]


def _base(profile: str) -> dict[str, object]:
    return {
        "profile": profile,
        "standard_version": "03/940r4",
        "carrier_frequency_hz": 5.25e9,
        "bandwidth_hz": 20e6,
        "num_rx": 1,
        "num_rx_antennas": 1,
        "num_tx": 1,
        "num_tx_antennas": 1,
        "environmental_speed_mps": 0.0,
        "num_sinusoids": 32,
        "spatial_quadrature_order": 64,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5,
        "line_of_sight": False,
        "large_scale_mode": "none",
        "fluorescent_effect": False,
        "moving_vehicle_effect": False,
        "normalize_profile_power": True,
    }


def _topology(execution: ExecutionContext, batch: int) -> TopologyBatch:
    backend = execution.array_backend
    tx = backend.zeros((batch, 1, 3), dtype=execution.float_dtype)
    tx[..., 2] = 3.0
    rx = backend.zeros((batch, 1, 3), dtype=execution.float_dtype)
    rx[..., 0] = 10.0
    rx[..., 2] = 1.5
    metadata = RealizationMetadata(
        "wireless", "wlan_statistical_topology", 1, execution.backend,
        execution.precision, execution.device, "numpy:74000:root", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx, rx,
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1), dtype="bool"),
        backend.zeros((batch, 1), dtype="int64"),
        metadata,
    )


def _assert_pdp(model, *, topology: bool = False) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 1_024
    sampling = ChannelSamplingRequest(batch, 1, 1_000.0, execution, execution.random_stream(74_000))
    request = WLANChannelSamplingRequest(
        sampling, _topology(execution, batch) if topology else None
    )
    cir = model.realize(request)
    measured = np.mean(np.abs(as_numpy(cir.coefficients)[:, 0, 0, 0, 0, :, 0]) ** 2, axis=0)
    expected = np.asarray([tap.power_linear for tap in load_ieee80211_profile(model.config.profile).taps])
    tolerance = 6.0 / np.sqrt(batch)
    assert np.all(np.abs(measured - expected) <= tolerance * np.maximum(expected, 0.002) + 8e-4)
    assert np.isclose(np.sum(measured), 1.0, atol=0.055)


@pytest.mark.tgn
@pytest.mark.parametrize("profile", list("ABCDEF"))
def test_every_tgn_profile_has_correct_ensemble_power_delay_profile(profile: str) -> None:
    model = TGnChannel(TGnConfig(**_base(profile)))
    _assert_pdp(model)
    expected_delays = load_ieee80211_profile(profile).delays_s
    assert model.diagnostics(
        WLANChannelSamplingRequest(
            ChannelSamplingRequest(
                1, 1, 1.0, ExecutionContext("numpy", "double"),
                ExecutionContext("numpy", "double").random_stream(1),
            )
        )
    ).tap_delays_s == expected_delays


@pytest.mark.tgac
def test_tgac_ensemble_pdp_is_preserved_by_bandwidth_extension() -> None:
    values = _base("D")
    values.update(standard_version="09/0308r12", bandwidth_hz=160e6, num_users=1, direction="downlink")
    _assert_pdp(TGacChannel(TGacConfig(**values)))


@pytest.mark.tgah
def test_tgah_ensemble_pdp_is_preserved_by_sub_ghz_extension() -> None:
    values = _base("B")
    values.update(
        standard_version="11/0968r4",
        carrier_frequency_hz=900e6,
        bandwidth_hz=8e6,
        link_type="ap_to_station",
        deployment="indoor",
    )
    _assert_pdp(TGahChannel(TGahConfig(**values)))


@pytest.mark.tgax
@pytest.mark.parametrize("scenario", ["indoor_residential", "indoor_enterprise", "outdoor_umi", "outdoor_uma"])
def test_tgax_each_scenario_preserves_small_scale_ensemble_pdp(scenario: str) -> None:
    values = _base("C")
    values.update(
        standard_version="14/0882r4",
        bandwidth_hz=80e6,
        scenario=scenario,
        wall_count=0,
        floor_count=0,
        fast_object_effect=False,
        street_width_m=20.0,
        average_building_height_m=5.0,
    )
    _assert_pdp(TGaxChannel(TGaxConfig(**values)), topology=True)
