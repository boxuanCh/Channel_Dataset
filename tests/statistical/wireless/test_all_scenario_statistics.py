from __future__ import annotations

from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    ScenarioConfig,
    TopologyBatch,
    create_scenario,
    load_large_scale_parameters,
)
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_bernoulli_rate,
    assert_correlation_matrix,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = pytest.mark.statistical

_SETTINGS = {
    "RMa": (2.0e9, 35.0, 1.5, 1_000.0, False),
    "UMa": (3.5e9, 25.0, 1.5, 100.0, False),
    "UMi": (3.5e9, 10.0, 1.5, 100.0, False),
    "InH": (6.0e9, 3.0, 1.5, 20.0, True),
    "InF": (28.0e9, 8.0, 1.5, 50.0, True),
}


def _scenario(name: str):
    frequency, *_ = _SETTINGS[name]
    if name == "InH":
        return create_scenario(ScenarioConfig(name, frequency, 120.0, 50.0, 3.0))
    if name == "InF":
        return create_scenario(ScenarioConfig(name, frequency, 100.0, 50.0, 10.0, 0.4, 5.0))
    return create_scenario(ScenarioConfig(name, frequency))


def _topology(name: str, batch_size: int, los: bool | None) -> TopologyBatch:
    execution = ExecutionContext("numpy", "double")
    backend = execution.array_backend
    _, tx_height, rx_height, distance, indoor = _SETTINGS[name]
    tx = np.tile(np.array([0.0, 0.0, tx_height]), (batch_size, 1, 1))
    rx = np.tile(np.array([distance, 0.0, rx_height]), (batch_size, 1, 1))
    metadata = RealizationMetadata(
        "wireless", "statistical_topology", 1, "numpy", "double", "cpu",
        "numpy:73000:root", "0" * 64, {}, "test",
    )
    los_mask = None if los is None else backend.full((batch_size, 1, 1), los, dtype="bool")
    return TopologyBatch(
        backend.array(tx, dtype="float64"),
        backend.array(rx, dtype="float64"),
        backend.zeros((batch_size, 1, 3), dtype="float64"),
        backend.zeros((batch_size, 1, 3), dtype="float64"),
        backend.zeros((batch_size, 1, 3), dtype="float64"),
        backend.zeros((batch_size, 1, 3), dtype="float64"),
        backend.full((batch_size, 1), indoor, dtype="bool"),
        backend.zeros((batch_size, 1), dtype="int64"),
        metadata,
        los_mask,
    )


def _log_domain_values(result) -> np.ndarray:
    degrees = 180.0 / pi
    return np.column_stack(
        (
            np.log10(as_numpy(result.delay_spread_s).reshape(-1)),
            np.log10(as_numpy(result.azimuth_spread_departure_rad).reshape(-1) * degrees),
            np.log10(as_numpy(result.azimuth_spread_arrival_rad).reshape(-1) * degrees),
            as_numpy(result.shadow_fading_db).reshape(-1),
            as_numpy(result.k_factor_db).reshape(-1),
            np.log10(as_numpy(result.zenith_spread_arrival_rad).reshape(-1) * degrees),
            np.log10(as_numpy(result.zenith_spread_departure_rad).reshape(-1) * degrees),
        )
    )


@pytest.mark.parametrize("scenario_name", list(_SETTINGS))
@pytest.mark.parametrize("state", ["los", "nlos"])
def test_all_scenario_lsp_marginals_and_cross_correlations(
    scenario_name: str, state: str
) -> None:
    execution = ExecutionContext("numpy", "double")
    scenario = _scenario(scenario_name)
    table = load_large_scale_parameters(scenario_name)
    statistics = table[state]
    result = scenario.sample_large_scale(
        _topology(scenario_name, 6_000, state == "los"),
        execution.random_stream(73_001),
    )
    values = _log_domain_values(result)
    means = np.asarray(statistics["means"], dtype=np.float64)
    standard_deviations = np.asarray(statistics["stddevs"], dtype=np.float64)
    for index, (mean, standard_deviation) in enumerate(zip(means, standard_deviations, strict=True)):
        if standard_deviation == 0.0:
            assert np.all(values[:, index] == mean)
        else:
            assert_mean(values[:, index], mean, standard_deviation)
            assert_standard_deviation(values[:, index], standard_deviation)
    active = standard_deviations > 0.0
    expected = np.asarray(table["cross_correlation_matrix"])[np.ix_(active, active)]
    assert_correlation_matrix(values[:, active], expected)


@pytest.mark.parametrize("scenario_name", list(_SETTINGS))
def test_all_scenario_los_sampling_rates_match_probability(scenario_name: str) -> None:
    execution = ExecutionContext("numpy", "double")
    scenario = _scenario(scenario_name)
    topology = _topology(scenario_name, 8_000, None)
    geometry = scenario.validate_topology(topology)
    expected = float(as_numpy(scenario.los_probability(geometry))[0, 0, 0])
    result = scenario.sample_large_scale(topology, execution.random_stream(73_002))
    # Every packaged NLOS state has exactly zero K, while LOS K is continuous.
    observed_los = as_numpy(result.k_factor_db).reshape(-1) != 0.0
    assert_bernoulli_rate(observed_los, expected)


@pytest.mark.parametrize("scenario_name", list(_SETTINGS))
def test_all_scenario_delay_spread_fields_match_exponential_decorrelation(
    scenario_name: str,
) -> None:
    execution = ExecutionContext("numpy", "double")
    backend = execution.array_backend
    _, tx_height, rx_height, base_distance, indoor = _SETTINGS[scenario_name]
    decorrelation = float(
        load_large_scale_parameters(scenario_name)["spatial_correlation_distances_m"][0]
    )
    batch = 4_000
    tx = backend.zeros((batch, 1, 3), dtype="float64")
    tx[..., 2] = tx_height
    rx_template = np.array(
        [
            [base_distance, 0.0, rx_height],
            [base_distance, decorrelation, rx_height],
            [base_distance, 2.0 * decorrelation, rx_height],
        ]
    )
    rx = backend.array(np.tile(rx_template[None, ...], (batch, 1, 1)), dtype="float64")
    metadata = RealizationMetadata(
        "wireless", "spatial_statistical_topology", 1, "numpy", "double", "cpu",
        "numpy:73003:root", "0" * 64, {}, "test",
    )
    topology = TopologyBatch(
        tx,
        rx,
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 3, 3), dtype="float64"),
        backend.zeros((batch, 1, 3), dtype="float64"),
        backend.zeros((batch, 3, 3), dtype="float64"),
        backend.full((batch, 3), indoor, dtype="bool"),
        backend.zeros((batch, 1), dtype="int64"),
        metadata,
        backend.ones((batch, 3, 1), dtype="bool"),
    )
    result = _scenario(scenario_name).sample_large_scale(
        topology, execution.random_stream(73_003)
    )
    values = np.log10(as_numpy(result.delay_spread_s)[..., 0])
    observed = np.corrcoef(values, rowvar=False)[0]
    expected = np.exp(-np.array((0.0, 1.0, 2.0)))
    assert np.max(np.abs(observed - expected)) < 0.06
