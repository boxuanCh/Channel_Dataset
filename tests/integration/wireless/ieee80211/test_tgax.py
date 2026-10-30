from __future__ import annotations

from math import log10

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest, TopologyBatch
from channel_dataset.channels.wireless.ieee80211 import (
    TGaxChannel,
    TGaxConfig,
    WLANChannelSamplingRequest,
)
from channel_dataset.channels.wireless.ieee80211.large_scale import tgax_path_loss_db
from channel_dataset.core import RealizationMetadata


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _config(scenario: str, **overrides) -> TGaxConfig:
    values = {
        "profile": "B", "standard_version": "14/0882r4",
        "carrier_frequency_hz": 5e9, "bandwidth_hz": 80e6,
        "num_rx": 1, "num_rx_antennas": 1, "num_tx": 1,
        "num_tx_antennas": 1, "environmental_speed_mps": 0.0,
        "num_sinusoids": 32, "spatial_quadrature_order": 64,
        "rx_antenna_spacing_wavelengths": 0.5,
        "tx_antenna_spacing_wavelengths": 0.5, "line_of_sight": True,
        "large_scale_mode": "path_loss", "fluorescent_effect": False,
        "moving_vehicle_effect": False, "normalize_profile_power": True,
        "scenario": scenario, "wall_count": 0, "floor_count": 0,
        "fast_object_effect": False, "street_width_m": 20.0,
        "average_building_height_m": 20.0,
    }
    values.update(overrides)
    return TGaxConfig(**values)


def _topology(execution, distance: float, tx_height: float, rx_height: float) -> TopologyBatch:
    backend = execution.array_backend
    horizontal = (distance**2 - (tx_height - rx_height) ** 2) ** 0.5
    tx = backend.array([[[0.0, 0.0, tx_height]]], dtype=execution.float_dtype)
    rx = backend.array([[[horizontal, 0.0, rx_height]]], dtype=execution.float_dtype)
    zeros = backend.zeros((1, 1, 3), dtype=execution.float_dtype)
    metadata = RealizationMetadata(
        "wireless", "test_topology", 1, execution.backend, execution.precision,
        execution.device, "topology", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx, rx, zeros, zeros, zeros, zeros,
        backend.array([[False]], dtype="bool"), backend.array([[0]], dtype="int64"),
        metadata, link_mask=backend.array([[[True]]], dtype="bool"),
    )


@pytest.mark.ieee80211
@pytest.mark.tgax
def test_tgax_umi_los_and_nlos_reference_equations(backend) -> None:
    distance = backend.array([100.0], dtype="float64")
    ap_height = backend.array([10.0], dtype="float64")
    station_height = backend.array([1.5], dtype="float64")
    los = tgax_path_loss_db(
        distance, 5e9, "outdoor_umi", True, backend,
        ap_height_m=ap_height, station_height_m=station_height,
    )
    expected_los = 28.0 + 22.0 * log10(100.0) + 20.0 * log10(5.0)
    assert float(_as_numpy(los)[0]) == pytest.approx(expected_los, abs=1e-10)
    nlos = tgax_path_loss_db(
        distance, 5e9, "outdoor_umi", False, backend,
        ap_height_m=ap_height, station_height_m=station_height,
    )
    expected_nlos = 22.7 + 36.7 * log10(100.0) + 26.0 * log10(5.0)
    assert float(_as_numpy(nlos)[0]) == pytest.approx(expected_nlos, abs=1e-10)


@pytest.mark.ieee80211
@pytest.mark.tgax
@pytest.mark.parametrize("scenario", ["indoor_residential", "indoor_enterprise", "outdoor_umi", "outdoor_uma"])
def test_tgax_claimed_scenarios_generate_valid_cir(execution_context, scenario: str) -> None:
    outdoor = scenario.startswith("outdoor")
    topology = _topology(execution_context, 100.0 if outdoor else 10.0, 10.0 if outdoor else 2.0, 1.5)
    sampling = ChannelSamplingRequest(
        1, 3, 2e3, execution_context, execution_context.random_stream(989)
    )
    cir = TGaxChannel(_config(scenario)).realize(
        WLANChannelSamplingRequest(sampling, topology)
    )
    assert cir.shape == (1, 1, 1, 1, 1, 9, 3)
    assert np.isfinite(_as_numpy(cir.coefficients)).all()


@pytest.mark.ieee80211
@pytest.mark.tgax
def test_tgax_penetration_uses_scenario_wall_and_nonlinear_floor_rules(execution_context) -> None:
    request = WLANChannelSamplingRequest(
        ChannelSamplingRequest(1, 1, 2e3, execution_context, execution_context.random_stream(1)),
        _topology(execution_context, 10.0, 2.0, 1.5),
    )
    residential = TGaxChannel(_config("indoor_residential", wall_count=2, floor_count=2))
    enterprise = TGaxChannel(_config("indoor_enterprise", wall_count=2, floor_count=2))
    count = 2
    floor_loss = 18.3 * count ** ((count + 2) / (count + 1) - 0.46)
    assert residential._penetration_loss_db(request) == pytest.approx(10.0 + floor_loss)
    assert enterprise._penetration_loss_db(request) == pytest.approx(14.0 + floor_loss)


@pytest.mark.ieee80211
@pytest.mark.tgax
def test_tgax_fast_object_mode_changes_only_selected_cluster_taps(execution_context) -> None:
    topology = _topology(execution_context, 100.0, 10.0, 1.5)
    sampling = ChannelSamplingRequest(
        1, 8, 5e3, execution_context, execution_context.random_stream(414)
    )
    slow = TGaxChannel(_config("outdoor_umi", large_scale_mode="none"))
    fast = TGaxChannel(_config("outdoor_umi", large_scale_mode="none", fast_object_effect=True))
    slow_cir = slow.realize(WLANChannelSamplingRequest(sampling, topology))
    replay = ChannelSamplingRequest(
        1, 8, 5e3, execution_context, execution_context.random_stream(414)
    )
    fast_cir = fast.realize(WLANChannelSamplingRequest(replay, topology))
    slow_values = _as_numpy(slow_cir.coefficients)
    fast_values = _as_numpy(fast_cir.coefficients)
    # Tap 0 belongs only to cluster 1; overlapping taps involving cluster 2 differ.
    assert not np.array_equal(slow_values[..., 0, :], fast_values[..., 0, :])
    assert not np.array_equal(slow_values[..., 2, :], fast_values[..., 2, :])
