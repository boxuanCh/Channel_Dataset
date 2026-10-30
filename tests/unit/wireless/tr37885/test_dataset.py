from __future__ import annotations

from pathlib import Path

import pytest

from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr37885 import (
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
    load_v2x_realization,
    save_v2x_realization,
)
from channel_dataset.core import ExecutionContext

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]


def _realization():
    execution = ExecutionContext("numpy", "double", "cpu")
    scene = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(num_vehicles=4)
    ).generate(V2XSceneSamplingRequest(1, 2, execution, execution.random_stream(21)))
    request = V2XChannelSamplingRequest(
        ChannelSamplingRequest(1, 4, 1000.0, execution, execution.random_stream(22)),
        scene, 0, (0,), (1,),
    )
    return TR37885UrbanCDLChannel(TR37885UrbanCDLConfig()).realize_with_diagnostics(request)


def test_realization_round_trip(tmp_path: Path):
    original = _realization()
    save_v2x_realization(tmp_path / "v2x", original)
    loaded = load_v2x_realization(tmp_path / "v2x", ExecutionContext("numpy", "double", "cpu"))
    assert loaded.cir.shape == original.cir.shape
    assert loaded.state.snapshot_index == 0
    assert loaded.link_conditions.condition_codes.shape == original.link_conditions.condition_codes.shape
    assert loaded.scene.shape == original.scene.shape
    assert loaded.scene.metadata.resolved_configuration == original.scene.metadata.resolved_configuration


@pytest.mark.torch
def test_realization_loads_on_another_backend(tmp_path: Path):
    pytest.importorskip("torch")
    root = save_v2x_realization(tmp_path / "v2x", _realization())
    loaded = load_v2x_realization(root, ExecutionContext("torch", "double", "cpu"))
    assert loaded.cir.metadata.backend == "torch"
    assert loaded.scene.metadata.backend == "torch"
    assert loaded.scene.shape == (1, 4, 2)


def test_corrupt_array_is_rejected(tmp_path: Path):
    root = save_v2x_realization(tmp_path / "v2x", _realization())
    path = root / "state.condition_codes.npy"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_v2x_realization(root, ExecutionContext("numpy", "double", "cpu"))
