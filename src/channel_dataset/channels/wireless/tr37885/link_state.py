"""Urban LOS, vehicle-obstructed NLOSv, and building-obstructed NLOS states."""

from __future__ import annotations

from math import exp

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr37885.grid import TR37885UrbanGrid
from channel_dataset.channels.wireless.tr37885.types import (
    INVALID,
    LOS,
    NLOS,
    NLOSV,
    V2XLinkConditionBatch,
    V2XSceneTrajectory,
)
from channel_dataset.core import RealizationMetadata


def urban_los_probability(distance_m: float) -> float:
    """Return the TR 37.885 urban same-street LOS probability."""

    if distance_m < 0.0:
        raise ValueError("distance_m must be nonnegative")
    return min(1.0, 1.05 * exp(-0.0114 * distance_m))


def classify_urban_links(
    scene: V2XSceneTrajectory,
    grid: TR37885UrbanGrid,
    tx_vehicle_indices: tuple[int, ...],
    rx_vehicle_indices: tuple[int, ...],
    random_stream,
    *,
    metadata: RealizationMetadata | None = None,
) -> V2XLinkConditionBatch:
    """Classify selected links for every mobility snapshot."""

    backend = get_backend(scene.metadata.backend)
    b_count, _, time_count = scene.shape
    codes = [[[[INVALID for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    same = [[[[False for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    obstructed = [[[[False for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    probabilities = [[[[0.0 for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    blocker_types = [[[[0 for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    blocker_heights = [[[[0.0 for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    episodes = [[[[0 for _ in range(time_count)] for _ in tx_vehicle_indices] for _ in rx_vehicle_indices] for _ in range(b_count)]
    type_probabilities = _fleet_probabilities(scene, backend)
    for batch in range(b_count):
        for rx_axis, rx_vehicle in enumerate(rx_vehicle_indices):
            for tx_axis, tx_vehicle in enumerate(tx_vehicle_indices):
                prior = INVALID
                episode = 0
                sampled_same_street = None
                for time in range(time_count):
                    active = bool(backend.to_scalar(scene.active_mask[batch, rx_vehicle, time])) and bool(
                        backend.to_scalar(scene.active_mask[batch, tx_vehicle, time])
                    ) and rx_vehicle != tx_vehicle
                    if not active:
                        codes[batch][rx_axis][tx_axis][time] = INVALID
                        prior = INVALID
                        continue
                    rx_x = float(backend.to_scalar(scene.positions_m[batch, rx_vehicle, time, 0]))
                    rx_y = float(backend.to_scalar(scene.positions_m[batch, rx_vehicle, time, 1]))
                    tx_x = float(backend.to_scalar(scene.positions_m[batch, tx_vehicle, time, 0]))
                    tx_y = float(backend.to_scalar(scene.positions_m[batch, tx_vehicle, time, 1]))
                    dx, dy = rx_x - tx_x, rx_y - tx_y
                    distance = (dx * dx + dy * dy) ** 0.5
                    rx_street = int(backend.to_scalar(scene.street_ids[batch, rx_vehicle, time]))
                    tx_street = int(backend.to_scalar(scene.street_ids[batch, tx_vehicle, time]))
                    on_same_street = rx_street >= 0 and rx_street == tx_street
                    blocked = (
                        not on_same_street
                        or grid.segment_intersects_building((tx_x, tx_y), (rx_x, rx_y))
                    )
                    p_los = urban_los_probability(distance)
                    probabilities[batch][rx_axis][tx_axis][time] = p_los
                    obstructed[batch][rx_axis][tx_axis][time] = blocked
                    same[batch][rx_axis][tx_axis][time] = on_same_street
                    if blocked:
                        current = NLOS
                        sampled_same_street = None
                    else:
                        if sampled_same_street is None:
                            if prior in {NLOS, INVALID}:
                                episode += 1
                            draw = backend.to_scalar(random_stream.fork(
                                f"tr37885.batch.{batch}.link.{min(rx_vehicle, tx_vehicle)}."
                                f"{max(rx_vehicle, tx_vehicle)}.episode.{episode}.condition"
                            ).uniform((1,), dtype=scene.metadata.precision == "single" and "float32" or "float64")[0])
                            sampled_same_street = LOS if float(draw) < p_los else NLOSV
                        current = sampled_same_street
                    codes[batch][rx_axis][tx_axis][time] = current
                    episodes[batch][rx_axis][tx_axis][time] = episode
                    if current == NLOSV:
                        blocker_type = _sample_blocker_type(
                            type_probabilities[batch], backend, random_stream,
                            batch, rx_vehicle, tx_vehicle, episode,
                        )
                        blocker_types[batch][rx_axis][tx_axis][time] = blocker_type
                        blocker_heights[batch][rx_axis][tx_axis][time] = (1.6, 1.6, 3.0)[blocker_type - 1]
                    prior = current
    device = scene.metadata.device
    return V2XLinkConditionBatch(
        backend.array(codes, dtype="int64", device=device),
        backend.array(same, dtype="bool", device=device),
        backend.array(obstructed, dtype="bool", device=device),
        backend.array(probabilities, dtype="float32" if scene.metadata.precision == "single" else "float64", device=device),
        backend.array(blocker_types, dtype="int64", device=device),
        backend.array(blocker_heights, dtype="float32" if scene.metadata.precision == "single" else "float64", device=device),
        backend.array(episodes, dtype="int64", device=device),
        metadata or scene.metadata,
    )


def _fleet_probabilities(scene, backend):
    result = []
    for batch in range(scene.shape[0]):
        counts = [0, 0, 0]
        for vehicle in range(scene.shape[1]):
            counts[int(backend.to_scalar(scene.vehicle_type_codes[batch, vehicle])) - 1] += 1
        total = sum(counts)
        result.append(tuple(value / total for value in counts))
    return result


def _sample_blocker_type(probabilities, backend, stream, batch, rx, tx, episode):
    value = float(backend.to_scalar(stream.fork(
        f"tr37885.link.{min(rx, tx)}.{max(rx, tx)}.episode.{episode}.blocker_type.{batch}"
    ).uniform((1,))[0]))
    return 1 if value < probabilities[0] else 2 if value < probabilities[0] + probabilities[1] else 3


__all__ = ["classify_urban_links", "urban_los_probability"]
