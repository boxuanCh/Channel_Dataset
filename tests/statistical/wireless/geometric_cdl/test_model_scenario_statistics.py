from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    GeometricCDLConfig,
    GeometricCDLModel,
    GeometricCDLSamplingRequest,
    ScenarioTimelineBatch,
    TopologyTrajectoryBatch,
    load_large_scale_parameters,
    load_ray_generation_parameters,
)
from channel_dataset.channels.wireless.tr38901 import load_cdl_profile
from channel_dataset.core import ExecutionContext, RealizationMetadata
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_correlation_matrix,
    assert_mean,
    assert_standard_deviation,
)

pytestmark = [pytest.mark.statistical, pytest.mark.geometric_cdl]

_SCENARIO_VARIANTS = (
    "RMa_LOS",
    "RMa_NLOS",
    "UMa_LOS",
    "UMa_NLOS",
    "UMi_LOS",
    "UMi_NLOS",
    "UMi_O2I_NLOS",
    "InH_Mixed_LOS",
    "InH_Open_NLOS",
    "InF_SL_LOS",
    "InF_DL_NLOS",
    "InF_SH_LOS",
    "InF_DH_NLOS",
)


def _metadata(execution: ExecutionContext, seed: int) -> RealizationMetadata:
    return RealizationMetadata(
        "wireless",
        "geometric_cdl_statistics",
        1,
        execution.backend,
        execution.precision,
        execution.device,
        f"{execution.backend}:{seed}:root",
        "0" * 64,
        {},
        "test",
    )


def _request(
    profiles: tuple[str, ...],
    *,
    seed: int,
    positions: np.ndarray | None = None,
) -> GeometricCDLSamplingRequest:
    execution = ExecutionContext("numpy", "double", "cpu")
    backend = execution.array_backend
    batch = len(profiles) if positions is None else len(positions)
    if positions is None:
        positions = np.arange(batch, dtype=np.float64)[:, None] * np.array(
            [137.0, 73.0, 29.0]
        )
    tx = np.zeros((batch, 1, 1, 3), dtype=np.float64)
    tx[:, 0, 0, :] = positions
    tx[..., 2] += 10.0
    rx = tx.copy()
    rx[..., 0] += 50.0
    rx[..., 2] -= 8.5
    zeros = backend.zeros((batch, 1, 1, 3), dtype="float64")
    metadata = _metadata(execution, seed)
    los = np.asarray(["NLOS" not in profiles[index % len(profiles)] for index in range(batch)])
    trajectory = TopologyTrajectoryBatch(
        backend.array([0.0], dtype="float64"),
        backend.array(tx, dtype="float64"),
        backend.array(rx, dtype="float64"),
        zeros,
        zeros,
        zeros,
        zeros,
        backend.arange(batch, dtype="int64")[:, None],
        backend.arange(batch, dtype="int64")[:, None] + 10_000,
        backend.zeros((batch, 1, 1), dtype="bool"),
        backend.arange(batch, dtype="int64")[:, None],
        metadata,
        los_mask=backend.array(los[:, None, None, None], dtype="bool"),
    )
    profile_map = {index: profile for index, profile in enumerate(profiles)}
    codes = np.arange(batch, dtype=np.int64) % len(profiles)
    timeline = ScenarioTimelineBatch(
        backend.array(codes[:, None, None, None], dtype="int64"),
        profile_map,
        metadata,
    )
    sampling = ChannelSamplingRequest(
        batch, 1, 1_000.0, execution, execution.random_stream(seed)
    )
    return GeometricCDLSamplingRequest(sampling, trajectory, timeline)


@pytest.mark.parametrize("standard_version", ["16.1.0", "19.2.0"])
def test_every_geometric_scenario_variant_selects_the_right_cluster_population(
    standard_version: str,
) -> None:
    request = _request(_SCENARIO_VARIANTS, seed=80_000)
    model = GeometricCDLModel(
        GeometricCDLConfig(
            standard_version,
            (3.5e9,),
            num_sinusoids=32,
            rays_per_cluster=4,
            solver_tolerance=1.0,
            solver_max_iterations=4,
            apply_path_loss=False,
            apply_shadow_fading=False,
        )
    )
    result = model.realize(request)
    ray_mask = as_numpy(result.carriers[0].rays.ray_mask)[..., 0]
    for batch, profile in enumerate(_SCENARIO_VARIANTS):
        family = profile.split("_")[0]
        if family == "InH":
            family = "InH"
        elif family == "InF":
            family = "InF"
        state = "nlos" if "NLOS" in profile else "los"
        expected = load_ray_generation_parameters(family)[state].num_clusters * 4
        assert np.count_nonzero(ray_mask[batch]) == expected
    assert np.all(np.isfinite(as_numpy(result.carriers[0].cir.coefficients)))
    assert np.all(as_numpy(result.carriers[0].cir.delays_s) >= 0.0)


@pytest.mark.parametrize("standard_version", ["16.1.0", "19.2.0"])
@pytest.mark.parametrize("profile_name", ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"])
def test_every_geometric_fixed_profile_preserves_reference_power_and_xpr(
    standard_version: str, profile_name: str
) -> None:
    request = _request(("UMi_LOS",), seed=80_001)
    model = GeometricCDLModel(
        GeometricCDLConfig(
            standard_version,
            (3.5e9,),
            initialization_mode="fixed_profile",
            fixed_profile=profile_name,
            num_sinusoids=16,
            rays_per_cluster=4,
            solver_tolerance=1.0,
            solver_max_iterations=4,
        )
    )
    state = model.realize(request).state.carrier_states[0]
    profile = load_cdl_profile(profile_name)
    expected_power = np.power(10.0, np.asarray(profile.powers_db) / 10.0)
    expected_power /= expected_power.sum()
    assert np.allclose(as_numpy(state.cluster_power_linear)[0, 0, 0], expected_power)
    assert np.allclose(
        10.0 * np.log10(as_numpy(state.xpr_linear)[0, 0, 0]), profile.xpr_db
    )


@pytest.mark.parametrize("family", ["RMa", "UMa", "UMi", "InH", "InF"])
@pytest.mark.parametrize("state", ["los", "nlos"])
def test_geometric_lsp_fields_follow_each_scenario_marginal_and_correlation_table(
    family: str, state: str
) -> None:
    batch = 2_500
    coordinates = np.random.default_rng(80_002).uniform(-20_000.0, 20_000.0, (batch, 3))
    profile = f"{family}_{state.upper()}"
    request = _request((profile,), seed=80_002, positions=coordinates)
    model = GeometricCDLModel(
        GeometricCDLConfig(
            "16.1.0",
            (3.5e9,),
            num_sinusoids=128,
            stationarity_distance_m=15.0,
            apply_path_loss=False,
            apply_shadow_fading=True,
        )
    )
    metadata = model._metadata(request)
    fields = model._field_values(request, model._field_bank(request, metadata))
    los = request.trajectory.los_mask
    trajectory = model._large_scale(request, metadata, fields, los, 3.5e9)
    values = np.column_stack(
        (
            np.log10(as_numpy(trajectory.delay_spread_s).reshape(-1)),
            np.log10(np.rad2deg(as_numpy(trajectory.azimuth_spread_departure_rad)).reshape(-1)),
            np.log10(np.rad2deg(as_numpy(trajectory.azimuth_spread_arrival_rad)).reshape(-1)),
            as_numpy(trajectory.shadow_fading_db).reshape(-1),
            as_numpy(trajectory.k_factor_db).reshape(-1),
            np.log10(np.rad2deg(as_numpy(trajectory.zenith_spread_arrival_rad)).reshape(-1)),
            np.log10(np.rad2deg(as_numpy(trajectory.zenith_spread_departure_rad)).reshape(-1)),
        )
    )
    resource = load_large_scale_parameters(family)
    statistics = resource[state]
    means = np.asarray(statistics["means"])
    deviations = np.asarray(statistics["stddevs"])
    for index, (mean, deviation) in enumerate(zip(means, deviations, strict=True)):
        if deviation == 0.0:
            assert np.allclose(values[:, index], mean)
        else:
            assert_mean(values[:, index], mean, deviation, z=7.0, absolute_floor=0.04)
            assert_standard_deviation(values[:, index], deviation, relative_floor=0.09)
    active = deviations > 0.0
    expected = np.asarray(resource["cross_correlation_matrix"])[np.ix_(active, active)]
    assert_correlation_matrix(values[:, active], expected, absolute_floor=0.09)
