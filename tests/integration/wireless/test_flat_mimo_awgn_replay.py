from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.fading import register_flat_mimo_model
from channel_dataset.channels.wireless.noise import AdditiveWhiteGaussianNoise, parse_awgn_config
from channel_dataset.channels.wireless.operators import apply_flat
from channel_dataset.config import ConfigEnvelope, load_config
from channel_dataset.registry import ModelRegistry


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def test_flat_mimo_application_and_awgn_replay(backend_name: str) -> None:
    root = Path(__file__).resolve().parents[3]
    channel_envelope = load_config(root / "configs" / "wireless" / "flat_mimo_fading.yaml")
    noise_envelope = load_config(root / "configs" / "wireless" / "awgn.yaml")
    registry = ModelRegistry()
    register_flat_mimo_model(registry)
    model = registry.create(channel_envelope)
    noise = AdditiveWhiteGaussianNoise(parse_awgn_config(noise_envelope.parameters))

    from channel_dataset.core import ExecutionContext

    execution = ExecutionContext(backend_name, "double", "cpu")

    def run():
        stream = execution.random_stream(2026)
        request = ChannelSamplingRequest(
            batch_size=8,
            num_time_samples=1,
            sampling_frequency_hz=1.0,
            execution=execution,
            random_stream=stream,
            request_id="flat-awgn-replay",
        )
        response = model.realize(request)
        signal = execution.array_backend.ones(
            (8, 1, 2, 16), dtype=execution.complex_dtype, device=execution.device
        )
        noiseless = apply_flat(response, signal)
        return response, noiseless, noise.apply(noiseless, stream)

    first_response, first_noiseless, first_noisy = run()
    second_response, second_noiseless, second_noisy = run()
    assert np.array_equal(_as_numpy(first_response.coefficients), _as_numpy(second_response.coefficients))
    assert np.array_equal(_as_numpy(first_noiseless), _as_numpy(second_noiseless))
    assert np.array_equal(_as_numpy(first_noisy), _as_numpy(second_noisy))
    assert not np.array_equal(_as_numpy(first_noisy), _as_numpy(first_noiseless))
    replay = ConfigEnvelope.from_realization_metadata(first_response.metadata)
    assert replay.model_digest == channel_envelope.model_digest
