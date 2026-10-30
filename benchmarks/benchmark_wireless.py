"""Non-blocking CPU benchmark for representative wireless operations."""

from __future__ import annotations

from time import perf_counter

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.fading import (
    RayleighBlockFading,
    RayleighBlockFadingConfig,
)
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import ExecutionContext


def benchmark(
    backend_name: str, iterations: int = 20, device: str = "cpu"
) -> float:
    """Return mean seconds for Rayleigh realization plus OFDM conversion."""

    execution = ExecutionContext(backend_name, "single", device)
    model = RayleighBlockFading(RayleighBlockFadingConfig(4, 4, 4, 4, 1.0))
    frequencies = subcarrier_frequencies(256, 15_000.0, execution)
    start = perf_counter()
    for index in range(iterations):
        request = ChannelSamplingRequest(
            32, 14, 15_000.0, execution, execution.random_stream(index)
        )
        cir_to_ofdm(model.realize(request), frequencies)
    if backend_name == "torch" and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    return (perf_counter() - start) / iterations


if __name__ == "__main__":
    for backend in ("torch", "numpy"):
        print(f"{backend}: {benchmark(backend):.6f} s/iteration")
