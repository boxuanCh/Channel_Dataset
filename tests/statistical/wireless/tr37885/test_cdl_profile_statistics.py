from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    LOS,
    NLOS,
    NLOSV,
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr37885.profiles import load_cdl_profile
from channel_dataset.core import ExecutionContext
from tests.statistical.statistical_assertions import as_numpy

pytestmark = [pytest.mark.statistical, pytest.mark.v2x, pytest.mark.tr37885]


def test_all_v2x_fixed_cdl_conditions_have_tabulated_ensemble_pdp() -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    batch = 1_024
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="B", num_vehicles=8)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(batch, 1, execution, execution.random_stream(77_000))
    )
    sampling = ChannelSamplingRequest(
        batch, 1, 2_000.0, execution, execution.random_stream(77_001)
    )
    model = TR37885UrbanCDLChannel(
        TR37885UrbanCDLConfig(
            apply_path_loss=False,
            apply_shadow_fading=False,
            apply_vehicle_blockage=False,
            moving_scatterers=False,
        )
    )
    # In the standardized traffic ordering, peer 1 supplies mostly LOS drops,
    # peer 2 is NLOSv, and peer 4 is on the perpendicular street (NLOS).
    realization = model.realize(V2XChannelSamplingRequest(sampling, scene, 0, (0,), (1, 2, 4)))
    coefficients = as_numpy(realization.cir.coefficients)[:, :, 0, 0, 0, :, 0]
    conditions = as_numpy(realization.link_conditions.condition_codes[..., 0])[:, :, 0]
    for code, name in ((LOS, "los"), (NLOSV, "nlosv"), (NLOS, "nlos")):
        selected = coefficients[conditions == code]
        assert selected.shape[0] >= 500
        profile = load_cdl_profile(name)
        expected = np.power(10.0, np.asarray(profile.powers_db) / 10.0)
        expected /= expected.sum()
        measured = np.mean(np.abs(selected[:, : expected.size]) ** 2, axis=0)
        tolerance = 6.0 / np.sqrt(selected.shape[0])
        assert np.all(
            np.abs(measured - expected)
            <= tolerance * np.maximum(expected, 0.002) + 8.0e-4
        )
