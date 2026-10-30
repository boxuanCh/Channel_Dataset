from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from channel_dataset.channels.wireless.fading import register_rayleigh_model
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.config import ConfigEnvelope, load_config
from channel_dataset.registry import ModelRegistry


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


@pytest.mark.parametrize(
    "config_name",
    ["rayleigh_block_fading.yaml", "rayleigh_block_fading_profile.yaml"],
)
def test_configured_rayleigh_ofdm_replays(backend_name: str, config_name: str) -> None:
    root = Path(__file__).resolve().parents[3]
    envelope = load_config(root / "configs" / "wireless" / config_name)
    registry = ModelRegistry()
    register_rayleigh_model(registry)
    model = registry.create(envelope)

    from channel_dataset.core import ExecutionContext

    execution = ExecutionContext(backend_name, "double", "cpu")

    def run():
        request = ChannelSamplingRequest(
            batch_size=2, num_time_samples=2, sampling_frequency_hz=1.0e6,
            execution=execution, random_stream=execution.random_stream(123), request_id="replay",
        )
        response = cir_to_ofdm(
            model.realize(request), subcarrier_frequencies(4, 15_000.0, execution)
        )
        signal = execution.array_backend.ones(
            (2, 1, 2, 2, 4), dtype=execution.complex_dtype, device="cpu"
        )
        return response, apply_ofdm(response, signal)

    first_response, first_output = run()
    second_response, second_output = run()
    assert first_response.metadata.config_digest == envelope.model_digest
    replay_envelope = ConfigEnvelope.from_realization_metadata(first_response.metadata)
    assert replay_envelope.to_document()["model"]["parameters"] == dict(envelope.parameters)
    assert np.array_equal(_as_numpy(first_response.coefficients), _as_numpy(second_response.coefficients))
    assert np.array_equal(_as_numpy(first_output), _as_numpy(second_output))
