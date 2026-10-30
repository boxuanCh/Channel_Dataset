from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.tr38901 import register_tdl_model
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.config import ConfigEnvelope, load_config
from channel_dataset.core import ExecutionContext
from channel_dataset.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[3]


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_configured_tdl_ofdm_replays(backend_name) -> None:
    envelope = load_config(ROOT / "configs" / "wireless" / "example_tdl.yaml")
    registry = ModelRegistry()
    register_tdl_model(registry)
    model = registry.create(envelope)
    execution = ExecutionContext(backend_name, "double")

    def run():
        request = ChannelSamplingRequest(
            2, 4, 30_720.0, execution, execution.random_stream(2026), "tdl-replay"
        )
        cir = model.realize(request)
        frequencies = subcarrier_frequencies(8, 15_000.0, execution)
        response = cir_to_ofdm(cir, frequencies)
        grid = execution.array_backend.ones((2, 1, 1, 4, 8), dtype="complex128")
        return cir, response, apply_ofdm(response, grid)

    first_cir, first_response, first_output = run()
    second_cir, second_response, second_output = run()
    assert np.array_equal(_as_numpy(first_cir.coefficients), _as_numpy(second_cir.coefficients))
    assert np.array_equal(
        _as_numpy(first_response.coefficients), _as_numpy(second_response.coefficients)
    )
    assert np.array_equal(_as_numpy(first_output), _as_numpy(second_output))
    replay = ConfigEnvelope.from_realization_metadata(first_cir.metadata)
    assert replay.model_digest == envelope.model_digest
