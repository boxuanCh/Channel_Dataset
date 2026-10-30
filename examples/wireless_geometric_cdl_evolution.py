"""Run a seeded multicarrier geometric-CDL trajectory example."""

from __future__ import annotations

from hashlib import sha256

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    GeometricCDLSamplingRequest,
    ScenarioTimelineBatch,
    TopologyTrajectoryBatch,
    cir_to_ofdm,
    register_geometric_cdl_model,
)
from channel_dataset.channels.wireless.transforms import subcarrier_frequencies
from channel_dataset.config import load_config
from channel_dataset.core import ExecutionContext, RealizationMetadata
from channel_dataset.registry import ModelRegistry


def main() -> None:
    """Generate a moving channel, a segment transition, and OFDM responses."""

    execution = ExecutionContext("numpy", "double", "cpu")
    backend = execution.array_backend
    metadata = RealizationMetadata(
        "wireless",
        "geometric_cdl_example_input",
        1,
        execution.backend,
        execution.precision,
        execution.device,
        "geometric-cdl-example-input",
        sha256(b"geometric-cdl-example-input").hexdigest(),
        {},
        "example",
    )
    count = 6
    times_s = backend.arange(count, dtype="float64") * 1.0e-6
    tx_positions_m = backend.zeros((1, 1, count, 3), dtype="float64")
    tx_positions_m[..., 2] = 10.0
    rx_positions_m = backend.zeros((1, 1, count, 3), dtype="float64")
    rx_positions_m[..., 0] = 50.0 + backend.arange(count, dtype="float64") * 0.001
    rx_positions_m[..., 2] = 1.5
    zeros = backend.zeros((1, 1, count, 3), dtype="float64")
    trajectory = TopologyTrajectoryBatch(
        times_s,
        tx_positions_m,
        rx_positions_m,
        zeros,
        zeros,
        zeros,
        zeros,
        backend.array([[100]], dtype="int64"),
        backend.array([[200]], dtype="int64"),
        backend.zeros((1, 1, count), dtype="bool"),
        backend.array([[10]], dtype="int64"),
        metadata,
    )
    scenarios = ScenarioTimelineBatch(
        backend.array([[[[0, 0, 0, 1, 1, 1]]]], dtype="int64"),
        {0: "UMi_LOS", 1: "UMi_NLOS"},
        metadata,
    )
    sampling = ChannelSamplingRequest(
        1,
        count,
        1.0e6,
        execution,
        execution.random_stream(20260925),
        "geometric-cdl-example",
    )

    registry = ModelRegistry()
    register_geometric_cdl_model(registry)
    model = registry.create(
        load_config("configs/wireless/geometric_cdl_multiband.yaml")
    )
    realization = model.realize(
        GeometricCDLSamplingRequest(
            sampling, trajectory, scenarios, diagnostics=True
        )
    )
    frequencies = subcarrier_frequencies(64, 15_000.0, execution)
    responses = tuple(
        cir_to_ofdm(carrier.cir, frequencies)
        for carrier in realization.carriers
    )

    print(
        "carriers (GHz):",
        [carrier.carrier_frequency_hz / 1.0e9 for carrier in realization.carriers],
    )
    print("segment boundaries:", realization.diagnostics["segment_boundaries"])
    print("dynamic CIR shapes:", [carrier.cir.shape for carrier in realization.carriers])
    print("OFDM response shapes:", [response.shape for response in responses])
    print("reusable state final time (s):", realization.state.last_time_s)


if __name__ == "__main__":
    main()
