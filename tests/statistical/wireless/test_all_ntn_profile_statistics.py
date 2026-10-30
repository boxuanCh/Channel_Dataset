from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    EARTH_RADIUS_M,
    NTNChannelSamplingRequest,
    NTNTopologyBatch,
    TR38811CDLChannel,
    TR38811CDLConfig,
    TR38811TDLChannel,
    TR38811TDLConfig,
    load_cdl_profile,
    load_tdl_profile,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import as_numpy

pytestmark = [pytest.mark.statistical, pytest.mark.ntn, pytest.mark.tr38811]


def _topology(execution: ExecutionContext, batch: int) -> NTNTopologyBatch:
    backend = execution.array_backend
    tx = backend.array(
        [[[[EARTH_RADIUS_M + 600_000.0, 0.0, 0.0]]]] * batch,
        dtype=execution.float_dtype,
    )
    rx = backend.array(
        [[[[EARTH_RADIUS_M, 0.0, 0.0]]]] * batch,
        dtype=execution.float_dtype,
    )
    zero = backend.zeros((batch, 1, 1, 3), dtype=execution.float_dtype)
    metadata = RealizationMetadata(
        "wireless", "ntn_statistical_topology", 1, execution.backend,
        execution.precision, execution.device, "numpy:75000:root", "0" * 64, {}, "test",
    )
    return NTNTopologyBatch(
        tx, rx, zero, zero, zero, zero, "satellite", "ground",
        backend.zeros((batch, 1), dtype="bool"), metadata,
    )


@pytest.mark.parametrize("profile_name", ["NTN-TDL-A", "NTN-TDL-B", "NTN-TDL-C", "NTN-TDL-D"])
def test_every_ntn_tdl_profile_has_correct_ensemble_tap_power(profile_name: str) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 4_096
    config = TR38811TDLConfig(
        profile_name, "15.4.0", "D3", "rural", 2.0e9, 20.0e6,
        100e-9, 1, 1, num_sinusoids=32, apply_large_scale=False,
    )
    sampling = ChannelSamplingRequest(batch, 1, 1_000.0, execution, execution.random_stream(75_001))
    cir = TR38811TDLChannel(config).realize(
        NTNChannelSamplingRequest(sampling, _topology(execution, batch))
    ).cir
    measured = np.mean(np.abs(as_numpy(cir.coefficients)[:, 0, 0, 0, 0, :, 0]) ** 2, axis=0)
    profile = load_tdl_profile(profile_name)
    expected = np.power(10.0, np.asarray(profile.powers_db) / 10.0)
    expected /= expected.sum()
    tolerance = 5.0 / np.sqrt(batch)
    assert np.all(np.abs(measured - expected) <= tolerance * expected + 5e-4)
    assert np.allclose(as_numpy(cir.delays_s)[0, 0, 0], np.asarray(profile.delays) * 100e-9)


@pytest.mark.parametrize("profile_name", ["NTN-CDL-A", "NTN-CDL-B", "NTN-CDL-C", "NTN-CDL-D"])
def test_every_ntn_cdl_profile_has_correct_ensemble_cluster_power(profile_name: str) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 768
    profile = load_cdl_profile(profile_name)
    config = TR38811CDLConfig(
        profile_name, "15.4.0", "D3", "urban", 2.0e9, 20.0e6,
        100e-9, max(profile.asa_degrees, 1.0), max(profile.zsa_degrees, 1.0),
        "profiles/wireless/antenna_isotropic_siso.yaml",
        "profiles/wireless/antenna_isotropic_siso.yaml",
        apply_large_scale=False,
    )
    sampling = ChannelSamplingRequest(batch, 1, 1_000.0, execution, execution.random_stream(75_002))
    cir = TR38811CDLChannel(config).realize(
        NTNChannelSamplingRequest(sampling, _topology(execution, batch))
    ).cir
    measured = np.mean(np.abs(as_numpy(cir.coefficients)[:, 0, 0, 0, 0, :, 0]) ** 2, axis=0)
    expected = np.power(10.0, np.asarray(profile.powers_db) / 10.0)
    expected /= expected.sum()
    tolerance = 6.0 / np.sqrt(batch)
    assert np.all(np.abs(measured - expected) <= tolerance * np.maximum(expected, 0.002) + 8e-4)
    assert np.allclose(as_numpy(cir.delays_s)[0, 0, 0], np.asarray(profile.delays) * 100e-9)
