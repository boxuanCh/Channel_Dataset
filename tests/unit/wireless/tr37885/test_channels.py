from __future__ import annotations

from dataclasses import replace

import pytest

from channel_dataset.channels.wireless import (
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    TR37885UrbanSystemChannel,
    TR37885UrbanSystemConfig,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
    register_tr37885_models,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ExecutionContext
from channel_dataset.registry import ModelRegistry

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]


def _request(backend_name="numpy", snapshot=0, state=None):
    execution = ExecutionContext(backend_name, "double", "cpu")
    simulator = TR37885UrbanGridSimulator(traffic=TR37885TrafficConfig(num_vehicles=6))
    scene = simulator.generate(V2XSceneSamplingRequest(1, 3, execution, execution.random_stream(10)))
    sampling = ChannelSamplingRequest(1, 16, 2000.0, execution, execution.random_stream(11))
    return V2XChannelSamplingRequest(sampling, scene, snapshot, (0, 1), (2, 3), state)


@pytest.mark.parametrize("backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)])
@pytest.mark.parametrize("kind", ["system", "cdl"])
def test_channel_shape_dtype_replay_and_finite(backend_name, kind):
    pytest.importorskip(backend_name if backend_name == "torch" else "numpy")
    request = _request(backend_name)
    model = (
        TR37885UrbanSystemChannel(TR37885UrbanSystemConfig())
        if kind == "system" else TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    )
    first = model.realize(request)
    second = model.realize(_request(backend_name))
    backend = request.sampling.execution.array_backend
    assert first.cir.shape[:5] == (1, 2, 1, 2, 1)
    assert first.cir.shape[-1] == 16
    assert backend.all_finite(first.cir.coefficients)
    assert backend.to_scalar(backend.max(backend.abs(first.cir.coefficients - second.cir.coefficients))) == 0.0


def test_sequence_carries_state():
    request = _request()
    simulator = TR37885UrbanGridSimulator(traffic=TR37885TrafficConfig(num_vehicles=6))
    model = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    results = simulator.realize_sequence(
        model, request.sampling, request.scene,
        request.tx_vehicle_indices, request.rx_vehicle_indices,
    )
    assert [result.state.snapshot_index for result in results] == [0, 1, 2]


def test_registry_constructs_both_models():
    registry = ModelRegistry()
    register_tr37885_models(registry)
    assert registry.create(ConfigEnvelope(1, "wireless", "tr37885_urban_system", {})).name == "tr37885_urban_system"
    assert registry.create(ConfigEnvelope(1, "wireless", "tr37885_urban_cdl", {})).name == "tr37885_urban_cdl"


def test_physical_configuration_hash_is_stable_and_sensitive():
    first = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    replay = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    changed = TR37885UrbanCDLChannel(replace(
        TR37885UrbanCDLConfig(), bandwidth_hz=10.0e6
    ))
    request = _request()
    assert first.realize(request).cir.metadata.config_digest == (
        replay.realize(_request()).cir.metadata.config_digest
    )
    assert first.realize(request).cir.metadata.config_digest != (
        changed.realize(_request()).cir.metadata.config_digest
    )


def test_unknown_config_field_is_rejected():
    registry = ModelRegistry()
    register_tr37885_models(registry)
    with pytest.raises(ValueError, match="unknown"):
        registry.create(ConfigEnvelope(1, "wireless", "tr37885_urban_system", {"typo": 1}))


def test_link_subset_does_not_change_existing_realization():
    full_request = _request()
    model = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    full = model.realize(full_request)
    subset = model.realize(V2XChannelSamplingRequest(
        full_request.sampling,
        full_request.scene,
        0,
        (full_request.tx_vehicle_indices[0],),
        (full_request.rx_vehicle_indices[0],),
    ))
    backend = full_request.sampling.execution.array_backend
    difference = (
        full.cir.coefficients[:, 0:1, :, 0:1]
        - subset.cir.coefficients
    )
    assert backend.to_scalar(backend.max(backend.abs(difference))) == 0.0


def test_stationary_episode_is_continuous_between_snapshots():
    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="B", num_vehicles=32)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(1, 2, execution, execution.random_stream(101))
    )
    sampling = ChannelSamplingRequest(
        1, 1, 2000.0, execution, execution.random_stream(102)
    )
    model = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig(
        apply_path_loss=False,
        apply_shadow_fading=False,
        apply_vehicle_blockage=False,
        moving_scatterers=False,
    ))
    first = model.realize(V2XChannelSamplingRequest(
        sampling, scene, 0, (16,), (17,)
    ))
    second = model.realize(V2XChannelSamplingRequest(
        sampling, scene, 1, (16,), (17,), first.state
    ))
    backend = execution.array_backend
    difference = first.cir.coefficients - second.cir.coefficients
    assert backend.to_scalar(backend.max(backend.abs(difference))) == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_self_link_selection_is_rejected():
    request = _request()
    with pytest.raises(ValueError, match="self-links"):
        V2XChannelSamplingRequest(
            request.sampling, request.scene, 0, (0,), (0,)
        )


def test_carried_state_requires_same_stream():
    request = _request()
    model = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig())
    first = model.realize(request)
    other_sampling = ChannelSamplingRequest(
        1, 16, 2000.0, request.sampling.execution,
        request.sampling.execution.random_stream(999),
    )
    with pytest.raises(ValueError, match="same random stream"):
        model.realize(V2XChannelSamplingRequest(
            other_sampling, request.scene, 1, (0, 1), (2, 3), first.state
        ))


@pytest.mark.torch
def test_torch_gradient_flows_through_geometry_and_path_loss():
    torch = pytest.importorskip("torch")
    execution = ExecutionContext("torch", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(num_vehicles=4)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(1, 1, execution, execution.random_stream(201))
    )
    scene.positions_m.requires_grad_(True)
    sampling = ChannelSamplingRequest(
        1, 2, 2000.0, execution, execution.random_stream(202)
    )
    model = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig(
        apply_shadow_fading=False,
        apply_vehicle_blockage=False,
        moving_scatterers=False,
    ))
    realization = model.realize(
        V2XChannelSamplingRequest(sampling, scene, 0, (0,), (1,))
    )
    power = torch.sum(torch.abs(realization.cir.coefficients) ** 2)
    power.backward()
    assert scene.positions_m.grad is not None
    assert torch.isfinite(scene.positions_m.grad).all()
    assert torch.count_nonzero(scene.positions_m.grad) > 0
