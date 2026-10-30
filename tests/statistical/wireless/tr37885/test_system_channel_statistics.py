from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    LOS,
    NLOS,
    NLOSV,
    TR37885TrafficConfig,
    TR37885UrbanGridSimulator,
    TR37885UrbanSystemChannel,
    TR37885UrbanSystemConfig,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.core import ExecutionContext
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = [pytest.mark.statistical, pytest.mark.v2x, pytest.mark.tr37885]


def test_all_v2x_system_conditions_have_correct_delay_k_and_power_statistics() -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    batch = 1_024
    scene = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="B", num_vehicles=8)
    ).generate(V2XSceneSamplingRequest(batch, 1, execution, execution.random_stream(79_000)))
    sampling = ChannelSamplingRequest(
        batch, 1, 2_000.0, execution, execution.random_stream(79_001)
    )
    model = TR37885UrbanSystemChannel(
        TR37885UrbanSystemConfig(
            apply_path_loss=False,
            apply_shadow_fading=False,
            apply_vehicle_blockage=False,
            moving_scatterers=False,
        )
    )
    result = model.realize(
        V2XChannelSamplingRequest(sampling, scene, 0, (0,), (1, 2, 4))
    )
    coefficients = as_numpy(result.cir.coefficients)[:, :, 0, 0, 0, :, 0]
    delays = as_numpy(result.cir.delays_s)[:, :, 0]
    masks = as_numpy(result.cir.path_mask)[:, :, 0]
    conditions = as_numpy(result.link_conditions.condition_codes[..., 0])[:, :, 0]

    settings = {
        LOS: (12, 3.48, 2.0),
        NLOSV: (19, 0.0, 4.5),
        NLOS: (19, None, None),
    }
    for code, (path_count, k_mean_db, k_std_db) in settings.items():
        selected = conditions == code
        assert np.count_nonzero(selected) >= 500
        selected_masks = masks[selected]
        selected_delays = delays[selected]
        selected_coefficients = coefficients[selected]
        assert np.all(np.sum(selected_masks, axis=-1) == path_count)
        assert np.all(selected_delays[:, 0] == 0.0)
        assert np.all(np.diff(selected_delays[:, :path_count], axis=-1) >= 0.0)

        total_power = np.sum(np.abs(selected_coefficients[:, :path_count]) ** 2, axis=-1)
        assert_mean(total_power, 1.0, 1.0, z=6.0, absolute_floor=0.08)
        if k_mean_db is not None and k_std_db is not None:
            specular_power = np.abs(selected_coefficients[:, 0]) ** 2
            k_db = 10.0 * np.log10(specular_power / (1.0 - specular_power))
            assert_mean(k_db, k_mean_db, k_std_db)
            assert_standard_deviation(k_db, k_std_db)
