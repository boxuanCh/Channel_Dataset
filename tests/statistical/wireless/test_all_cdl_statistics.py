from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import (
    CDLConfig,
    ClusteredDelayLine,
    load_cdl_profile,
)
from channel_dataset.core import ExecutionContext
from tests.statistical.statistical_assertions import as_numpy

pytestmark = pytest.mark.statistical


def _config(profile: str) -> CDLConfig:
    return CDLConfig(
        profile=profile,
        standard_version="16.1.0",
        delay_spread_s=100e-9,
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=100e6,
        link_direction="downlink",
        base_station_array_profile="package://profiles/wireless/antenna_isotropic_siso.yaml",
        terminal_array_profile="package://profiles/wireless/antenna_isotropic_siso.yaml",
        base_station_orientation_rad=(0.0, 0.0, 0.0),
        terminal_orientation_rad=(0.0, 0.0, 0.0),
        base_station_velocity_mps=(0.0, 0.0, 0.0),
        terminal_velocity_mps=(0.0, 0.0, 0.0),
        link_distance_m=100.0,
        k_factor_db=None,
    )


@pytest.mark.parametrize("profile_name", ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"])
def test_every_cdl_profile_has_correct_ensemble_cluster_power(profile_name: str) -> None:
    execution = ExecutionContext("numpy", "double")
    model = ClusteredDelayLine(_config(profile_name))
    request = ChannelSamplingRequest(2_048, 1, 1.0, execution, execution.random_stream(72_000))
    cir = model.realize(request)
    measured = np.mean(
        np.abs(as_numpy(cir.coefficients)[:, 0, 0, 0, 0, :, 0]) ** 2,
        axis=0,
    )
    profile = load_cdl_profile(profile_name)
    diffuse = np.power(10.0, np.asarray(profile.powers_db) / 10.0)
    specular = 0.0 if profile.specular_power_db is None else 10.0 ** (profile.specular_power_db / 10.0)
    expected = diffuse / (np.sum(diffuse) + specular)
    expected[0] += specular / (np.sum(diffuse) + specular)
    tolerance = 6.0 / np.sqrt(request.batch_size)
    assert np.all(np.abs(measured - expected) <= tolerance * np.maximum(expected, 0.002) + 5e-4)
    assert np.isclose(np.sum(measured), 1.0, atol=0.035)


@pytest.mark.parametrize("profile_name", ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"])
def test_every_cdl_profile_ray_cloud_preserves_cluster_means(profile_name: str) -> None:
    execution = ExecutionContext("numpy", "double")
    model = ClusteredDelayLine(_config(profile_name))
    request = ChannelSamplingRequest(64, 1, 1.0, execution, execution.random_stream(72_001))
    rays = model.sample_rays(request)
    profile = load_cdl_profile(profile_name)
    angle_pairs = (
        (rays.azimuth_arrival_rad, profile.aoa_degrees),
        (rays.azimuth_departure_rad, profile.aod_degrees),
        (rays.zenith_arrival_rad, profile.zoa_degrees),
        (rays.zenith_departure_rad, profile.zod_degrees),
    )
    for values, means_degrees in angle_pairs:
        samples = as_numpy(values)[:, 0, 0]
        # Circular mean is required at the azimuth wrap and is harmless for zenith.
        observed = np.angle(np.mean(np.exp(1j * samples), axis=(0, 2)))
        expected = np.deg2rad(means_degrees)
        difference = np.angle(np.exp(1j * (observed - expected)))
        assert np.max(np.abs(difference)) < 1.0e-12
    assert np.allclose(as_numpy(rays.xpr_linear), 10.0 ** (profile.xpr_db / 10.0))


@pytest.mark.parametrize("profile_name", ["CDL-D", "CDL-E"])
def test_los_cdl_first_cluster_has_tabulated_specular_to_diffuse_ratio(
    profile_name: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    model = ClusteredDelayLine(_config(profile_name))
    request = ChannelSamplingRequest(4_096, 1, 1.0, execution, execution.random_stream(72_002))
    first = as_numpy(model.realize(request).coefficients)[:, 0, 0, 0, 0, 0, 0]
    observed = np.abs(np.mean(first)) ** 2 / np.var(first)
    profile = load_cdl_profile(profile_name)
    expected = 10.0 ** ((profile.specular_power_db - profile.powers_db[0]) / 10.0)
    assert observed == pytest.approx(expected, rel=0.10)


def test_cdl_initial_ray_phases_are_uniform_and_uncorrelated() -> None:
    execution = ExecutionContext("numpy", "double")
    model = ClusteredDelayLine(_config("CDL-A"))
    request = ChannelSamplingRequest(256, 1, 1.0, execution, execution.random_stream(72_003))
    phase = as_numpy(model.sample_rays(request).initial_phase_rad).reshape(-1)
    phasor = np.exp(1j * phase)
    assert abs(np.mean(phasor)) < 0.02
    assert abs(np.mean(phasor[:-1].conj() * phasor[1:])) < 0.02
