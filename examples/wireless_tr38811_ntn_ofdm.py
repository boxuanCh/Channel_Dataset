"""Run a seeded NGSO NTN-TDL realization through an OFDM resource grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    NTNChannelSamplingRequest,
    NTNTopologyBatch,
    ntn_to_ofdm,
    register_tr38811_models,
)
from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.tr38811 import (
    EARTH_RADIUS_M,
    circular_equatorial_orbit,
)
from channel_dataset.channels.wireless.transforms import subcarrier_frequencies
from channel_dataset.config import load_config
from channel_dataset.core import RealizationMetadata
from channel_dataset.registry import ModelRegistry


def main() -> None:
    """Construct one NGSO-to-ground link, sample it, and apply it to OFDM."""

    root = Path(__file__).resolve().parents[1]
    envelope = load_config(root / "configs/wireless/tr38811/ngso_s_tdl.yaml")
    if envelope.execution is None:
        raise ValueError("the example requires an execution section")
    execution = envelope.execution.context()
    backend = execution.array_backend
    samples = 8

    platform_position, platform_velocity = circular_equatorial_orbit(
        600_000.0, 0.0, samples, 15_360_000.0, execution
    )
    tx_position = platform_position[None, None, :, :]
    tx_velocity = platform_velocity[None, None, :, :]
    rx_position = backend.array(
        [[[[EARTH_RADIUS_M, 0.0, 0.0]]]],
        dtype=execution.float_dtype,
        device=execution.device,
    )
    zeros_tx = backend.zeros(
        backend.shape(tx_position), dtype=execution.float_dtype, device=execution.device
    )
    zeros_rx = backend.zeros(
        backend.shape(rx_position), dtype=execution.float_dtype, device=execution.device
    )
    topology_metadata = RealizationMetadata(
        "wireless", "tr38811_topology", 1, execution.backend,
        execution.precision, execution.device, "deterministic-example",
        envelope.model_digest, {}, "0.1.0", standard_version="15.4.0",
    )
    topology = NTNTopologyBatch(
        tx_position,
        rx_position,
        tx_velocity,
        zeros_rx,
        zeros_tx,
        zeros_rx,
        "satellite",
        "ground",
        backend.zeros((1, 1), dtype="bool", device=execution.device),
        topology_metadata,
    )

    registry = ModelRegistry()
    register_tr38811_models(registry)
    model = registry.create(envelope)
    sampling = ChannelSamplingRequest(
        1,
        samples,
        15_360_000.0,
        execution,
        execution.random_stream(envelope.execution.seed),
        request_id="tr38811-ngso-ofdm-example",
    )
    realization = model.realize(NTNChannelSamplingRequest(sampling, topology))
    frequencies = subcarrier_frequencies(16, 15_000.0, execution)
    response = ntn_to_ofdm(realization, frequencies, include_propagation_delay=True)
    transmitted = backend.array(
        np.ones((1, 1, 1, samples, 16), dtype=execution.complex_dtype),
        dtype=execution.complex_dtype,
        device=execution.device,
    )
    received = apply_ofdm(response, transmitted)

    print(f"CIR shape: {realization.cir.shape}")
    print(f"OFDM response shape: {response.shape}")
    print(f"Received grid shape: {backend.shape(received)}")
    print(f"Initial range: {float(realization.geometry.slant_range_m[0, 0, 0, 0]):.1f} m")
    print(f"Configuration digest: {realization.cir.metadata.config_digest}")


if __name__ == "__main__":
    main()
