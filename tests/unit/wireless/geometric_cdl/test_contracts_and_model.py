from __future__ import annotations

import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    DynamicCIRBatch,
    GeometricCDLConfig,
    GeometricCDLModel,
    GeometricCDLSamplingRequest,
    ScenarioTimelineBatch,
    TopologyTrajectoryBatch,
    cir_to_ofdm,
    cir_to_time,
    load_geometric_cdl_realization,
    save_geometric_cdl_realization,
)


def _request(
    execution_context,
    realization_metadata,
    *,
    carriers=(3.5e9,),
    times_s=(0.0, 1.0e-6),
    rx_x_m=None,
    scenario_codes=None,
    prior_state=None,
    explicit_segment_boundaries=None,
):
    backend = execution_context.array_backend
    count = len(times_s)
    if rx_x_m is None:
        rx_x_m = (50.0,) * count
    times = backend.array(times_s, dtype=execution_context.float_dtype)
    tx = backend.array(
        [[[[0.0, 0.0, 10.0]] * count]], dtype=execution_context.float_dtype
    )
    rx = backend.array(
        [[[[value, 0.0, 1.5] for value in rx_x_m]]],
        dtype=execution_context.float_dtype,
    )
    zeros_tx = backend.zeros((1, 1, count, 3), dtype=execution_context.float_dtype)
    zeros_rx = backend.zeros((1, 1, count, 3), dtype=execution_context.float_dtype)
    trajectory = TopologyTrajectoryBatch(
        times, tx, rx, zeros_tx, zeros_rx, zeros_tx, zeros_rx,
        backend.zeros((1, 1), dtype="int64"),
        backend.ones((1, 1), dtype="int64"),
        backend.zeros((1, 1, count), dtype="bool"),
        backend.zeros((1, 1), dtype="int64"),
        realization_metadata,
    )
    if scenario_codes is None:
        scenario_codes = (0,) * count
    scenarios = ScenarioTimelineBatch(
        backend.array(
            [[[list(scenario_codes)]]], dtype="int64", device=execution_context.device
        ),
        {0: "UMi_LOS", 1: "UMi_NLOS"}, realization_metadata,
    )
    sampling = ChannelSamplingRequest(
        1, count, 1.0e6, execution_context, execution_context.random_stream(123)
    )
    return GeometricCDLSamplingRequest(
        sampling,
        trajectory,
        scenarios,
        prior_state,
        explicit_segment_boundaries,
    ), carriers


def test_trajectory_snapshot_preserves_common_axes(execution_context, realization_metadata):
    request, _ = _request(execution_context, realization_metadata)
    snapshot = request.trajectory.snapshot(1)
    assert snapshot.shape == (1, 1, 1)
    assert snapshot.tx_positions_m.shape == (1, 1, 3)


def test_dynamic_cir_snapshot_and_transforms(execution_context, realization_metadata):
    backend = execution_context.array_backend
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 2), dtype=execution_context.complex_dtype)
    delays = backend.array([[[[[0.0, 0.25e-6]]]]], dtype=execution_context.float_dtype)
    dynamic = DynamicCIRBatch(coefficients, delays, realization_metadata)
    assert dynamic.at_time(1).delays_s.shape == (1, 1, 1, 1)
    frequencies = backend.array([0.0, 1.0e6], dtype=execution_context.float_dtype)
    assert cir_to_ofdm(dynamic, frequencies).shape[-2:] == (2, 2)
    assert cir_to_time(dynamic, 1.0e6, lag_min=0, lag_max=2).shape[-2:] == (2, 3)


@pytest.mark.parametrize("standard_version", ["16.1.0", "19.2.0"])
def test_stationary_multicarrier_realization(
    execution_context, realization_metadata, standard_version
):
    request, carriers = _request(
        execution_context, realization_metadata, carriers=(3.5e9, 28.0e9)
    )
    model = GeometricCDLModel(
        GeometricCDLConfig(
            standard_version=standard_version,
            carrier_frequencies_hz=carriers,
            num_sinusoids=8,
            rays_per_cluster=4,
            solver_max_iterations=4,
            solver_tolerance=1.0,
        )
    )
    realization = model.realize(request)
    assert tuple(value.carrier_frequency_hz for value in realization.carriers) == carriers
    assert all(isinstance(value.cir, DynamicCIRBatch) for value in realization.carriers)
    assert all(value.cir.shape[-1] == 2 for value in realization.carriers)
    assert realization.state.last_time_s == pytest.approx(1.0e-6)
    backend = execution_context.array_backend
    assert float(backend.to_scalar(backend.max(backend.abs(
        realization.carriers[0].rays.fbs_positions_m
        - realization.carriers[1].rays.fbs_positions_m
    )))) < 1.0e-12


def test_invalid_prior_state_is_rejected(execution_context, realization_metadata):
    request, carriers = _request(execution_context, realization_metadata)
    first = GeometricCDLModel(
        GeometricCDLConfig("16.1.0", carriers, num_sinusoids=8, rays_per_cluster=4)
    ).realize(request)
    second = GeometricCDLModel(
        GeometricCDLConfig("16.1.0", (2.0e9,), num_sinusoids=8, rays_per_cluster=4)
    )
    with pytest.raises(ValueError):
        second.realize(
            GeometricCDLSamplingRequest(
                request.sampling, request.trajectory, request.scenarios, first.state
            )
        )


def test_persistence_round_trip(
    execution_context, realization_metadata, tmp_path
):
    request, carriers = _request(execution_context, realization_metadata)
    model = GeometricCDLModel(
        GeometricCDLConfig(
            "16.1.0", carriers, num_sinusoids=8, rays_per_cluster=4,
            solver_max_iterations=4, solver_tolerance=1.0,
        )
    )
    expected = model.realize(request)
    save_geometric_cdl_realization(tmp_path / "geometric", expected)
    actual = load_geometric_cdl_realization(tmp_path / "geometric", execution_context)
    backend = execution_context.array_backend
    error = backend.max(backend.abs(
        expected.carriers[0].cir.coefficients - actual.carriers[0].cir.coefficients
    ))
    assert float(backend.to_scalar(error)) == pytest.approx(0.0)
    assert actual.state.model_digest == expected.state.model_digest


def test_resumed_chunks_equal_one_shot(execution_context, realization_metadata):
    config = GeometricCDLConfig(
        "16.1.0", (3.5e9,), num_sinusoids=8, rays_per_cluster=4,
        solver_max_iterations=4, solver_tolerance=1.0,
    )
    model = GeometricCDLModel(config)
    full_request, _ = _request(
        execution_context,
        realization_metadata,
        times_s=(0.0, 1.0e-6, 2.0e-6, 3.0e-6),
        rx_x_m=(50.0, 50.001, 50.002, 50.003),
    )
    first_request, _ = _request(
        execution_context, realization_metadata, rx_x_m=(50.0, 50.001)
    )
    first = model.realize(first_request)
    second_request, _ = _request(
        execution_context,
        realization_metadata,
        times_s=(2.0e-6, 3.0e-6),
        rx_x_m=(50.002, 50.003),
        prior_state=first.state,
    )
    full = model.realize(full_request)
    second = model.realize(second_request)
    backend = execution_context.array_backend
    error = backend.max(backend.abs(
        full.carriers[0].cir.coefficients[..., 2:]
        - second.carriers[0].cir.coefficients
    ))
    assert float(backend.to_scalar(error)) < 1.0e-11


def test_explicit_segment_transition_uses_two_generations(
    execution_context, realization_metadata
):
    request, carriers = _request(
        execution_context,
        realization_metadata,
        times_s=(0.0, 1.0e-6, 2.0e-6, 3.0e-6),
        explicit_segment_boundaries=(2,),
    )
    model = GeometricCDLModel(
        GeometricCDLConfig(
            "16.1.0", carriers, num_sinusoids=8, rays_per_cluster=4,
            solver_max_iterations=4, solver_tolerance=1.0,
        )
    )
    realization = model.realize(request)
    rays = realization.carriers[0].rays
    backend = execution_context.array_backend
    clusters_per_generation = realization.state.carrier_states[0].cluster_ids.shape[-1]
    first_active = backend.max(
        backend.max(
            backend.astype(
                rays.amplitude_weight[..., :clusters_per_generation, :, :] > 0.0,
                "int64",
            ),
            axis=4,
        ),
        axis=3,
    )
    second_active = backend.max(
        backend.max(
            backend.astype(
                rays.amplitude_weight[..., clusters_per_generation:, :, :] > 0.0,
                "int64",
            ),
            axis=4,
        ),
        axis=3,
    )
    active_generations = first_active + second_active
    assert int(backend.to_scalar(backend.max(active_generations))) <= 2
    assert rays.shape[3] == 2 * clusters_per_generation


def test_scenario_boundary_between_chunks_matches_one_shot(
    execution_context, realization_metadata
):
    config = GeometricCDLConfig(
        "16.1.0", (3.5e9,), num_sinusoids=8, rays_per_cluster=4,
        solver_max_iterations=4, solver_tolerance=1.0,
    )
    model = GeometricCDLModel(config)
    full_request, _ = _request(
        execution_context,
        realization_metadata,
        times_s=(0.0, 1.0e-6, 2.0e-6, 3.0e-6),
        scenario_codes=(0, 0, 1, 1),
    )
    first_request, _ = _request(execution_context, realization_metadata)
    first = model.realize(first_request)
    second_request, _ = _request(
        execution_context,
        realization_metadata,
        times_s=(2.0e-6, 3.0e-6),
        scenario_codes=(1, 1),
        prior_state=first.state,
    )
    full = model.realize(full_request)
    second = model.realize(second_request)
    backend = execution_context.array_backend
    error = backend.max(backend.abs(
        full.carriers[0].cir.coefficients[..., 2:]
        - second.carriers[0].cir.coefficients
    ))
    assert float(backend.to_scalar(error)) < 1.0e-11


def test_torch_geometry_and_phase_are_differentiable(
    execution_context, realization_metadata
):
    if execution_context.backend != "torch":
        pytest.skip("PyTorch-only gradient contract")
    request, carriers = _request(execution_context, realization_metadata)
    request.trajectory.tx_positions_m.requires_grad_(True)
    request.trajectory.rx_positions_m.requires_grad_(True)
    model = GeometricCDLModel(
        GeometricCDLConfig(
            "16.1.0", carriers, num_sinusoids=8, rays_per_cluster=4,
            solver_max_iterations=4, solver_tolerance=1.0,
            apply_path_loss=False, apply_shadow_fading=False,
        )
    )
    coefficients = model.realize(request).carriers[0].cir.coefficients
    coefficients.abs().sum().backward()
    for positions in (
        request.trajectory.tx_positions_m,
        request.trajectory.rx_positions_m,
    ):
        assert positions.grad is not None
        assert bool(positions.grad.isfinite().all())
        assert float(positions.grad.abs().max()) > 0.0
