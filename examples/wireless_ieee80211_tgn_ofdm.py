"""Seeded TGn Model B to OFDM resource-grid example."""

from pathlib import Path

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211 import (
    WLANChannelSamplingRequest,
    register_ieee80211_models,
)
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def main() -> None:
    """Generate and apply a replayable TGn MIMO OFDM response."""

    root = Path(__file__).resolve().parents[1]
    envelope = load_config(root / "configs/wireless/ieee80211/tgn_model_b.yaml")
    registry = ModelRegistry()
    register_ieee80211_models(registry)
    model = registry.create(envelope)
    execution = envelope.execution.context()
    sampling = ChannelSamplingRequest(
        batch_size=1,
        num_time_samples=4,
        sampling_frequency_hz=20_000.0,
        execution=execution,
        random_stream=execution.random_stream(envelope.execution.seed),
        request_id="tgn-ofdm-example",
    )
    cir = model.realize(WLANChannelSamplingRequest(sampling))
    frequencies = subcarrier_frequencies(64, 312_500.0, execution)
    response = cir_to_ofdm(cir, frequencies)
    grid = execution.array_backend.ones(
        (1, 1, 2, 4, 64), dtype=execution.complex_dtype, device=execution.device
    )
    received = apply_ofdm(response, grid)
    print("CIR:", cir.shape, "OFDM:", response.shape, "received:", received.shape)
    print("standard:", cir.metadata.standard_version, "seed:", cir.metadata.root_seed)


if __name__ == "__main__":
    main()
