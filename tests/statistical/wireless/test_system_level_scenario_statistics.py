from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    ScenarioConfig,
    SystemLevelWirelessChannel,
    TopologyBatch,
    load_ray_generation_parameters,
)
from channel_dataset.channels.wireless.antennas import load_antenna_layout
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = pytest.mark.statistical

_SCENARIOS = {
    "RMa": (ScenarioConfig("RMa", 2.0e9), 35.0, 1.5, 1_000.0, False),
    "UMa": (ScenarioConfig("UMa", 3.5e9), 25.0, 1.5, 100.0, False),
    "UMi": (ScenarioConfig("UMi", 3.5e9), 10.0, 1.5, 100.0, False),
    "InH": (ScenarioConfig("InH", 6.0e9, 120.0, 50.0, 3.0), 3.0, 1.5, 20.0, True),
    "InF": (
        ScenarioConfig("InF", 28.0e9, 100.0, 50.0, 10.0, 0.4, 5.0),
        8.0,
        1.5,
        50.0,
        True,
    ),
}


def _topology(
    execution: ExecutionContext,
    scenario_name: str,
    batch: int,
    los: bool,
) -> TopologyBatch:
    _, tx_height, rx_height, distance, indoor = _SCENARIOS[scenario_name]
    backend = execution.array_backend
    tx = backend.zeros((batch, 1, 3), dtype=execution.float_dtype)
    tx[..., 2] = tx_height
    rx = backend.zeros((batch, 1, 3), dtype=execution.float_dtype)
    rx[..., 0] = distance
    rx[..., 2] = rx_height
    metadata = RealizationMetadata(
        "wireless", "system_statistical_topology", 1, execution.backend,
        execution.precision, execution.device, "numpy:78000:root", "0" * 64, {}, "test",
    )
    return TopologyBatch(
        tx,
        rx,
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.zeros((batch, 1, 3), dtype=execution.float_dtype),
        backend.full((batch, 1), indoor, dtype="bool"),
        backend.zeros((batch, 1), dtype="int64"),
        metadata,
        backend.full((batch, 1, 1), los, dtype="bool"),
    )


@pytest.mark.parametrize("scenario_name", list(_SCENARIOS))
@pytest.mark.parametrize("state", ["los", "nlos"])
def test_every_system_level_scenario_has_correct_ray_and_cir_statistics(
    scenario_name: str,
    state: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    batch = 768
    topology = _topology(execution, scenario_name, batch, state == "los")
    layout = load_antenna_layout("profiles/wireless/antenna_isotropic_siso.yaml")
    model = SystemLevelWirelessChannel(_SCENARIOS[scenario_name][0], topology, layout, layout)
    request = ChannelSamplingRequest(
        batch, 1, 1_000.0, execution, execution.random_stream(78_001)
    )
    result = model.realize_with_diagnostics(request)
    rays = result.rays
    parameters = load_ray_generation_parameters(scenario_name)[state]
    cluster_mask = as_numpy(rays.clusters.cluster_mask)
    ray_mask = as_numpy(rays.ray_mask)

    assert np.all(np.sum(cluster_mask, axis=-1) == parameters.num_clusters)
    assert np.all(np.sum(ray_mask, axis=(-2, -1)) == parameters.num_clusters * 20)
    powers = as_numpy(rays.clusters.powers_linear)
    assert np.allclose(np.sum(powers, axis=-1), 1.0, atol=2.0e-12)
    delays = as_numpy(rays.clusters.delays_s)
    assert np.all(delays[..., 0] == 0.0)
    assert np.all(np.diff(delays[..., : parameters.num_clusters], axis=-1) >= 0.0)

    xpr_db = 10.0 * np.log10(as_numpy(rays.xpr_linear)[ray_mask])
    assert_mean(xpr_db, parameters.xpr_mean_db, parameters.xpr_std_db)
    assert_standard_deviation(xpr_db, parameters.xpr_std_db)
    phase = as_numpy(rays.initial_phase_rad)[ray_mask]
    assert abs(np.mean(np.exp(1j * phase))) < 0.012

    attenuation = np.power(
        10.0,
        -(as_numpy(result.large_scale.path_loss_db) + as_numpy(result.large_scale.shadow_fading_db))
        / 20.0,
    )
    coefficients = as_numpy(result.cir.coefficients)[:, 0, 0, 0, 0, :, 0]
    coefficients /= attenuation[:, 0, 0, None]
    total_power = np.sum(np.abs(coefficients) ** 2, axis=-1)
    assert_mean(total_power, 1.0, 1.0, z=6.0, absolute_floor=0.08)
