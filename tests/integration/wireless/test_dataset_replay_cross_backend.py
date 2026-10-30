import importlib.util

import pytest

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    ReplayChannel,
    load_cir,
    save_cir,
)
from channel_dataset.channels.wireless.fading import (
    RayleighBlockFading,
    RayleighBlockFadingConfig,
)
from channel_dataset.core import ExecutionContext


@pytest.mark.torch
@pytest.mark.parametrize(("source", "target"), [("numpy", "torch"), ("torch", "numpy")])
def test_generate_serialize_cross_backend_and_replay(tmp_path, source, target):
    if source == "torch" and importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    source_execution = ExecutionContext(source, "double", "cpu")
    source_request = ChannelSamplingRequest(
        1, 2, 1.0, source_execution, source_execution.random_stream(91)
    )
    model = RayleighBlockFading(RayleighBlockFadingConfig(1, 1, 1, 1, 1.0))
    original = model.realize(source_request)
    destination = save_cir(tmp_path / f"{source}-realization", original)
    target_execution = ExecutionContext(target, "double", "cpu")
    loaded = load_cir(destination, target_execution)
    target_request = ChannelSamplingRequest(
        1, 2, 1.0, target_execution, target_execution.random_stream(91)
    )
    replayed = ReplayChannel((loaded,)).realize(target_request)
    assert replayed.shape == original.shape
    assert replayed.metadata.config_digest == original.metadata.config_digest
