"""Generate, store, load, and replay a config-driven Rayleigh CIR."""

from pathlib import Path

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    ReplayChannel,
    load_cir,
    save_cir,
)
from channel_dataset.channels.wireless.fading import register_rayleigh_model
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def main() -> None:
    """Run the dataset round trip using the packaged example configuration."""

    envelope = load_config("configs/wireless/rayleigh_block_fading.yaml")
    if envelope.execution is None:
        raise ValueError("example configuration requires execution settings")
    execution = envelope.execution.context()
    registry = ModelRegistry()
    register_rayleigh_model(registry)
    model = registry.create(envelope)
    request = ChannelSamplingRequest(
        2, 8, 15_000.0, execution, execution.random_stream(envelope.execution.seed)
    )
    destination = Path("rayleigh-cir-example")
    save_cir(destination, model.realize(request))
    stored = load_cir(destination, execution)
    replayed = ReplayChannel((stored,)).realize(request)
    print(replayed.shape)


if __name__ == "__main__":
    main()
