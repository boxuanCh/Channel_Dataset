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
    load_tgah_path_loss_profile,
    load_tgax_outdoor_profile,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = [pytest.mark.statistical, pytest.mark.ieee80211]


def _base(profile: str, los: bool, mode: str) -> dict[str, object]:
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
        "line_of_sight": los,
        "large_scale_mode": mode,
        "fluorescent_effect": False,
        "moving_vehicle_effect": False,
        "normalize_profile_power": True,
    }


def _model(kind: str, profile: str, los: bool, mode: str, variant: str):
    values = _base(profile, los, mode)
    if kind == "tgn":
        return TGnChannel(TGnConfig(**values))
    if kind == "tgac":
        values.update(
            standard_version="09/0308r12",
            bandwidth_hz=160e6,
            num_users=1,
            direction="downlink",
        )
        return TGacChannel(TGacConfig(**values))
    if kind == "tgah":
        deployment, link_type = variant.split(":")
        values.update(
            standard_version="11/0968r4",
            carrier_frequency_hz=900e6,
            bandwidth_hz=8e6,
            deployment=deployment,
            link_type=link_type,
        )
        return TGahChannel(TGahConfig(**values))
    values.update(
        standard_version="14/0882r4",
        bandwidth_hz=80e6,
        scenario=variant,
        wall_count=2 if variant.startswith("indoor") else 0,
        floor_count=1 if variant.startswith("indoor") else 0,
        fast_object_effect=False,
        street_width_m=20.0,
        average_building_height_m=5.0,
    )
    return TGaxChannel(TGaxConfig(**values))


def _shadow_std(kind: str, profile: str, los: bool, variant: str) -> float:
    if kind in {"tgn", "tgac"} or (kind == "tgax" and variant.startswith("indoor")):
        resource = load_ieee80211_profile(profile)
        return resource.shadow_std_los_db if los else resource.shadow_std_nlos_db
    if kind == "tgah":
        return load_tgah_path_loss_profile().shadow_std_db[variant.split(":")[0]]
    resource = load_tgax_outdoor_profile().scenarios[variant]
    return float(resource["shadow_std_los_db" if los else "shadow_std_nlos_db"])


def _topology(execution: ExecutionContext, batch: int) -> TopologyBatch:
    backend = execution.array_backend
    tx = backend.zeros((batch, 1, 3), dtype="float64")
    tx[..., 2] = 10.0
    rx = backend.zeros((batch, 1, 3), dtype="float64")
    rx[..., 0] = 100.0
    rx[..., 2] = 1.5
    metadata = RealizationMetadata(
        "wireless",
        "wlan_large_scale_statistics",
        1,
        "numpy",
        "double",
        "cpu",
        "numpy:81000:root",
        "0" * 64,
        {},
        "test",
    )
    return TopologyBatch(
        tx,
        rx,
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 1), dtype="bool"),
        backend.zeros((batch, 1), dtype="int64"),
        metadata,
    )


_CASES = [
    *((kind, profile, los, "") for kind in ("tgn", "tgac") for profile in "ABCDEF" for los in (True, False)),
    *(("tgah", "B", los, f"{deployment}:{link_type}")
      for deployment in ("indoor", "outdoor_macro", "outdoor_pico", "outdoor_hotspot")
      for link_type in ("ap_to_station", "station_to_station") for los in (True, False)),
    *(("tgax", "C", los, scenario)
      for scenario in ("indoor_residential", "indoor_enterprise", "outdoor_umi", "outdoor_uma")
      for los in (True, False)),
]


@pytest.mark.parametrize(("kind", "profile", "los", "variant"), _CASES)
def test_every_wlan_large_scale_variant_has_the_configured_shadow_distribution(
    kind: str, profile: str, los: bool, variant: str
) -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    batch = 3_000
    request = WLANChannelSamplingRequest(
        ChannelSamplingRequest(
            batch, 1, 1_000.0, execution, execution.random_stream(81_000)
        ),
        _topology(execution, batch),
    )
    path_only = as_numpy(
        _model(kind, profile, los, "path_loss", variant)._large_scale_amplitude(request)
    )
    shadowed = as_numpy(
        _model(kind, profile, los, "path_loss_shadowing", variant)._large_scale_amplitude(
            request
        )
    )
    assert np.all((path_only > 0.0) & (path_only < 1.0))
    shadow_db = -20.0 * np.log10(shadowed / path_only)
    expected_std = _shadow_std(kind, profile, los, variant)
    assert_mean(shadow_db, 0.0, expected_std)
    assert_standard_deviation(shadow_db, expected_std)
