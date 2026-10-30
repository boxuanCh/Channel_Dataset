"""Run a replayable Rayleigh block-fading OFDM example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_dataset.channels.wireless.fading import register_rayleigh_model
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    envelope = load_config(root / "configs" / "wireless" / "rayleigh_block_fading.yaml")
    if envelope.execution is None:
        raise ValueError("the example requires an execution section")

    registry = ModelRegistry()
    register_rayleigh_model(registry)
    model = registry.create(envelope)
    execution = envelope.execution.context()
    request = ChannelSamplingRequest(
        batch_size=2,
        num_time_samples=4,
        sampling_frequency_hz=15_360_000.0,
        execution=execution,
        random_stream=execution.random_stream(envelope.execution.seed),
        request_id="rayleigh-ofdm-example",
    )
    cir = model.realize(request)
    frequencies = subcarrier_frequencies(16, 15_000.0, execution)
    response = cir_to_ofdm(cir, frequencies)
    backend = execution.array_backend
    signal = backend.array(
        np.ones((2, 1, 2, 4, 16), dtype=execution.complex_dtype),
        dtype=execution.complex_dtype,
        device=execution.device,
    )
    received = apply_ofdm(response, signal)
    print(f"CIR shape: {cir.shape}")
    print(f"OFDM response shape: {response.shape}")
    print(f"Received grid shape: {backend.shape(received)}")
    print(f"Configuration digest: {cir.metadata.config_digest}")


if __name__ == "__main__":
    main()
