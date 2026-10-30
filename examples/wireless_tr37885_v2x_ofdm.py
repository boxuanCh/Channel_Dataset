"""Run a seeded TR 37.885 urban-grid V2X-to-OFDM example."""

from __future__ import annotations

from channel_dataset.channels.wireless import (
    TR37885TrafficConfig,
    TR37885UrbanCDLChannel,
    TR37885UrbanCDLConfig,
    TR37885UrbanGridSimulator,
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
)
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, subcarrier_frequencies
from channel_dataset.core import ExecutionContext


def main() -> None:
    """Generate a scene, one channel snapshot, and a received resource grid."""

    execution = ExecutionContext("numpy", "double", "cpu")
    simulator = TR37885UrbanGridSimulator(
        traffic=TR37885TrafficConfig(option="A", num_vehicles=16)
    )
    scene = simulator.generate(
        V2XSceneSamplingRequest(
            batch_size=1,
            num_mobility_steps=4,
            execution=execution,
            random_stream=execution.random_stream(37885),
            request_id="v2x-scene-example",
        )
    )
    sampling = ChannelSamplingRequest(
        batch_size=1,
        num_time_samples=8,
        sampling_frequency_hz=15_000.0,
        execution=execution,
        random_stream=execution.random_stream(9),
        request_id="v2x-channel-example",
    )
    request = V2XChannelSamplingRequest(
        sampling=sampling,
        scene=scene,
        snapshot_index=0,
        tx_vehicle_indices=(0, 1),
        rx_vehicle_indices=(2, 3),
    )
    realization = TR37885UrbanCDLChannel(TR37885UrbanCDLConfig()).realize(request)
    frequencies = subcarrier_frequencies(64, 15_000.0, execution)
    response = cir_to_ofdm(realization.cir, frequencies)
    backend = execution.array_backend
    signal = backend.ones(
        (1, 2, 1, 8, 64), dtype=execution.complex_dtype, device=execution.device
    )
    received = apply_ofdm(response, signal)

    print(f"scene shape: {scene.shape}")
    print(f"CIR shape: {realization.cir.shape}")
    print(f"OFDM response shape: {response.shape}")
    print(f"received grid shape: {backend.shape(received)}")
    print(f"standard baseline: {realization.diagnostics.values['standard_baseline']}")


if __name__ == "__main__":
    main()
