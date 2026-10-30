"""Segment detection and smooth cluster-generation transition weights."""

from __future__ import annotations

from math import pi, sin, sqrt
from typing import Any


def segment_amplitude_weights(
    boundaries: tuple[int, ...],
    time_count: int,
    overlap_fraction: float,
    backend: Any,
    *,
    dtype: str,
    device: str,
) -> Any:
    """Return generation amplitude weights on ``[generation, time]`` axes."""

    if time_count <= 0:
        raise ValueError("time_count must be positive")
    if any(value < 0 or value >= time_count for value in boundaries):
        raise ValueError("segment boundaries must lie inside the time range")
    starts = (0, *boundaries)
    ends = (*boundaries, time_count)
    weights = [[0.0] * time_count for _ in starts]
    for generation, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for time in range(start, end):
            weights[generation][time] = 1.0
    for transition, boundary in enumerate(boundaries):
        right_length = ends[transition + 1] - boundary
        overlap = max(1, round(overlap_fraction * right_length))
        overlap_end = min(ends[transition + 1], boundary + overlap + 1)
        count = overlap_end - boundary
        for offset, time in enumerate(range(boundary, overlap_end)):
            normalized = 1.0 if count == 1 else offset / (count - 1)
            ramp = 0.5 * (1.0 + sin(pi * (normalized - 0.5)))
            weights[transition][time] = sqrt(max(0.0, 1.0 - ramp))
            weights[transition + 1][time] = sqrt(max(0.0, ramp))
    return backend.array(weights, dtype=dtype, device=device)


__all__ = ["segment_amplitude_weights"]
