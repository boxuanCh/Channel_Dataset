from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    EARTH_RADIUS_M,
    NTNChannelSamplingRequest,
    NTNLinkGeometry,
    NTNTopologyBatch,
    TR38811DeploymentConfig,
    TR38811SystemChannel,
    TR38811SystemConfig,
    load_cdl_profile,
)
from channel_dataset.channels.wireless.tr38811.large_scale import (
    los_probability,
    sample_large_scale,
    sample_los,
)
from channel_dataset.channels.wireless.tr38811.profiles import load_resource
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_bernoulli_rate,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = [pytest.mark.statistical, pytest.mark.ntn, pytest.mark.tr38811]

_ELEVATIONS = np.arange(10.0, 100.0, 10.0)
_DEPLOYMENTS = {
    "D1": (35_786_000.0, "satellite", 20.0e9),
    "D2": (35_786_000.0, "satellite", 2.0e9),
    "D3": (600_000.0, "satellite", 2.0e9),
    "D4": (600_000.0, "satellite", 20.0e9),
    "D5": (20_000.0, "haps", 2.0e9),
}


def _metadata() -> RealizationMetadata:
    return RealizationMetadata(
        "wireless", "ntn_scenario_statistics", 1, "numpy", "double", "cpu",
        "numpy:80000:root", "0" * 64, {}, "test",
    )


def _synthetic_geometry(
    execution: ExecutionContext,
    batch: int,
) -> NTNLinkGeometry:
    backend = execution.array_backend
    shape = (batch, len(_ELEVATIONS), 1, 1)
    elevation = np.deg2rad(_ELEVATIONS)[None, :, None, None]
    elevation = backend.array(np.broadcast_to(elevation, shape), dtype="float64")
    zero = backend.zeros(shape, dtype="float64")
    slant = backend.full(shape, 1_000_000.0, dtype="float64")
    direction = backend.zeros((*shape, 3), dtype="float64")
    direction[..., 0] = 1.0
    return NTNLinkGeometry(
        slant,
        elevation,
        zero,
        zero,
        slant / 299_792_458.0,
        zero,
        zero,
        backend.ones(shape, dtype="bool"),
        direction,
        _metadata(),
    )


def _synthetic_topology(
    execution: ExecutionContext,
    batch: int,
    *,
    indoor: bool = False,
) -> NTNTopologyBatch:
    backend = execution.array_backend
    receiver_count = len(_ELEVATIONS)
    tx = backend.zeros((batch, 1, 1, 3), dtype="float64")
    tx[..., 0] = EARTH_RADIUS_M + 600_000.0
    rx = backend.zeros((batch, receiver_count, 1, 3), dtype="float64")
    rx[..., 0] = EARTH_RADIUS_M
    tx_zero = backend.zeros((batch, 1, 1, 3), dtype="float64")
    rx_zero = backend.zeros((batch, receiver_count, 1, 3), dtype="float64")
    return NTNTopologyBatch(
        tx,
        rx,
        tx_zero,
        rx_zero,
        tx_zero,
        rx_zero,
        "satellite",
        "ground",
        backend.full((batch, receiver_count), indoor, dtype="bool"),
        _metadata(),
    )


@pytest.mark.parametrize("environment", ["dense_urban", "urban", "suburban", "rural"])
def test_ntn_los_probability_and_sampling_for_every_environment_and_elevation(
    environment: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    table = np.asarray(load_resource("los_probability")[environment], dtype=np.float64)
    elevation = execution.array_backend.array(np.deg2rad(_ELEVATIONS), dtype="float64")
    assert np.array_equal(as_numpy(los_probability(elevation, environment, execution)), table)

    batch = 10_000
    sampled = sample_los(
        _synthetic_geometry(execution, batch),
        _synthetic_topology(execution, batch),
        environment,
        execution.random_stream(80_001),
        execution,
    )
    for index, probability in enumerate(table):
        assert_bernoulli_rate(as_numpy(sampled)[:, index, 0], float(probability))


@pytest.mark.parametrize("environment", ["dense_urban", "urban", "suburban", "rural"])
@pytest.mark.parametrize("band", ["s", "ka"])
@pytest.mark.parametrize("state", ["los", "nlos"])
def test_ntn_large_scale_statistics_for_every_environment_band_and_state(
    environment: str,
    band: str,
    state: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 4_000
    frequency = 2.0e9 if band == "s" else 20.0e9
    deployment = "D3" if band == "s" else "D4"
    geometry = _synthetic_geometry(execution, batch)
    topology = _synthetic_topology(execution, batch)
    los = state == "los"
    los_mask = execution.array_backend.full(
        (batch, len(_ELEVATIONS), 1), los, dtype="bool"
    )
    config = TR38811DeploymentConfig(
        deployment,
        "15.4.0",
        environment,
        frequency,
        20.0e6,
        apply_o2i_loss=False,
        apply_atmospheric_loss=False,
        apply_scintillation=False,
    )
    result = sample_large_scale(
        geometry,
        topology,
        los_mask,
        config,
        execution.random_stream(80_002),
        execution,
        _metadata(),
    )
    table = load_resource("large_scale")["environments"][environment][band]
    sigma = np.asarray(table[f"{state}_sigma"], dtype=np.float64)
    shadow = as_numpy(result.shadow_fading_db)[..., 0]
    for index, expected_std in enumerate(sigma):
        assert_mean(shadow[:, index, 0], 0.0, float(expected_std))
        assert_standard_deviation(shadow[:, index, 0], float(expected_std))
    expected_clutter = (
        np.zeros_like(sigma)
        if los
        else np.asarray(table["nlos_clutter"], dtype=np.float64)
    )
    assert np.allclose(as_numpy(result.clutter_db)[0, :, 0, 0], expected_clutter)
    assert np.allclose(
        as_numpy(result.total_db),
        as_numpy(result.free_space_db)
        + as_numpy(result.clutter_db)
        + as_numpy(result.shadow_fading_db),
    )


def _deployment_topology(
    execution: ExecutionContext,
    deployment: str,
    batch: int,
    los: bool,
) -> NTNTopologyBatch:
    backend = execution.array_backend
    altitude, kind, _ = _DEPLOYMENTS[deployment]
    tx = backend.zeros((batch, 1, 1, 3), dtype="float64")
    tx[..., 0] = EARTH_RADIUS_M + altitude
    rx = backend.zeros((batch, 1, 1, 3), dtype="float64")
    rx[..., 0] = EARTH_RADIUS_M
    zero = backend.zeros((batch, 1, 1, 3), dtype="float64")
    return NTNTopologyBatch(
        tx,
        rx,
        zero,
        zero,
        zero,
        zero,
        kind,
        "ground",
        backend.full((batch, 1), deployment == "D5", dtype="bool"),
        _metadata(),
        backend.full((batch, 1, 1), los, dtype="bool"),
    )


@pytest.mark.parametrize("deployment", list(_DEPLOYMENTS))
@pytest.mark.parametrize("state", ["los", "nlos"])
def test_ntn_system_channel_selects_correct_kernel_in_every_deployment(
    deployment: str,
    state: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 32
    _, _, frequency = _DEPLOYMENTS[deployment]
    config = TR38811SystemConfig(
        "15.4.0",
        deployment,
        "urban",
        frequency,
        20.0e6,
        "profiles/wireless/antenna_isotropic_siso.yaml",
        "profiles/wireless/antenna_isotropic_siso.yaml",
        100e-9,
        15.0,
        7.0,
        apply_large_scale=False,
    )
    sampling = ChannelSamplingRequest(
        batch, 1, 1_000.0, execution, execution.random_stream(80_003)
    )
    result = TR38811SystemChannel(config).realize(
        NTNChannelSamplingRequest(
            sampling,
            _deployment_topology(execution, deployment, batch, state == "los"),
        )
    )
    profile = load_cdl_profile("NTN-CDL-C" if state == "los" else "NTN-CDL-A")
    expected_delays = np.asarray(profile.delays) * 100e-9
    assert np.all(as_numpy(result.los_mask) == (state == "los"))
    assert np.allclose(as_numpy(result.cir.delays_s)[..., : expected_delays.size], expected_delays)
