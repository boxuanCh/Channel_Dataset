from __future__ import annotations

import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811 import EARTH_RADIUS_M, NTNTopologyBatch
from channel_dataset.core import ExecutionContext, RealizationMetadata


def metadata(execution: ExecutionContext) -> RealizationMetadata:
    return RealizationMetadata(
        "wireless", "ntn_topology", 1, execution.backend, execution.precision,
        execution.device, "test", "0" * 64, {}, "0.1.0",
    )


def zenith_topology(
    execution: ExecutionContext,
    *,
    batch_size: int = 1,
    altitude_m: float = 600_000.0,
    platform_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NTNTopologyBatch:
    backend = get_backend(execution.backend)
    tx_position = backend.array(
        [[[[EARTH_RADIUS_M + altitude_m, 0.0, 0.0]]]] * batch_size,
        dtype=execution.float_dtype,
        device=execution.device,
    )
    rx_position = backend.array(
        [[[[EARTH_RADIUS_M, 0.0, 0.0]]]] * batch_size,
        dtype=execution.float_dtype,
        device=execution.device,
    )
    tx_velocity = backend.array(
        [[[[*platform_velocity_mps]]]] * batch_size,
        dtype=execution.float_dtype,
        device=execution.device,
    )
    zero = backend.zeros(
        (batch_size, 1, 1, 3), dtype=execution.float_dtype, device=execution.device
    )
    indoor = backend.zeros((batch_size, 1), dtype="bool", device=execution.device)
    return NTNTopologyBatch(
        tx_position, rx_position, tx_velocity, zero, zero, zero,
        "satellite", "ground", indoor, metadata(execution),
    )


@pytest.fixture
def zenith_topology_factory():
    """Return a factory for deterministic zenith service-link topologies."""

    return zenith_topology


__all__ = ["metadata", "zenith_topology", "zenith_topology_factory"]
