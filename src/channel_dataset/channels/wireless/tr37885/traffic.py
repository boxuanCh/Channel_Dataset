"""Standardized vehicle dropping and urban-grid mobility."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from math import atan2, log, pi

from channel_dataset.channels.wireless.tr37885.config import TR37885TrafficConfig
from channel_dataset.channels.wireless.tr37885.grid import TR37885UrbanGrid, V2XLane
from channel_dataset.channels.wireless.tr37885.types import (
    V2XChannelSamplingRequest,
    V2XSceneSamplingRequest,
    V2XSceneTrajectory,
)
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import RealizationMetadata

_VEHICLES = {
    1: (5.0, 2.0, 1.6, 0.75),
    2: (5.0, 2.0, 1.6, 1.6),
    3: (13.0, 2.6, 3.0, 3.0),
}
_OPTION_B_SPEEDS_MPS = (60.0 / 3.6, 50.0 / 3.6, 25.0 / 3.6, 15.0 / 3.6)


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class TR37885UrbanGridSimulator:
    """Generate repeatable lane-constrained urban V2X trajectories."""

    def __init__(
        self,
        grid: TR37885UrbanGrid | None = None,
        traffic: TR37885TrafficConfig | None = None,
    ) -> None:
        self.grid = grid or TR37885UrbanGrid()
        self.traffic = traffic or TR37885TrafficConfig()
        if self.grid.config.mode != self.traffic.mode:
            raise ValueError("grid and traffic modes must match")

    def _metadata(self, request: V2XSceneSamplingRequest) -> RealizationMetadata:
        resolved = {"grid": asdict(self.grid.config), "traffic": asdict(self.traffic)}
        digest = ConfigEnvelope(1, "wireless", "tr37885_urban_scene", resolved).model_digest
        return RealizationMetadata(
            "wireless", "tr37885_urban_scene", 1,
            request.execution.backend, request.execution.precision, request.execution.device,
            request.random_stream.stream_id, digest, resolved, _software_version(),
            request.request_id, root_seed=request.random_stream.root_seed,
            model_version="1", standard_version="15.3.0",
        )

    def _vehicle_type(self, request: V2XSceneSamplingRequest, batch: int, vehicle: int) -> int:
        probabilities = self.traffic.resolved_vehicle_type_probabilities
        value = request.execution.array_backend.to_scalar(
            request.random_stream.fork(f"tr37885.vehicle.{batch}.{vehicle}.type").uniform(
                (1,), dtype=request.execution.float_dtype
            )[0]
        )
        return 1 if value < probabilities[0] else 2 if value < probabilities[0] + probabilities[1] else 3

    def _speed(self, lane: V2XLane) -> float:
        if self.traffic.option == "A":
            return 60.0 / 3.6 * self.traffic.speed_multiplier
        if lane.orientation == "vertical":
            return 0.0
        slot = lane.speed_slot + (0 if lane.direction > 0 else 2)
        return _OPTION_B_SPEEDS_MPS[slot] * self.traffic.speed_multiplier

    def generate(self, request: V2XSceneSamplingRequest) -> V2XSceneTrajectory:
        """Generate one complete mobility trajectory."""

        execution, backend = request.execution, request.execution.array_backend
        batches_positions: list[list[list[list[float]]]] = []
        batches_velocities: list[list[list[list[float]]]] = []
        batches_orientations: list[list[list[list[float]]]] = []
        batches_active: list[list[list[bool]]] = []
        batches_lanes: list[list[list[int]]] = []
        batches_streets: list[list[list[int]]] = []
        batches_types: list[list[int]] = []
        batches_dimensions: list[list[list[float]]] = []
        batches_antennas: list[list[float]] = []
        vehicle_ids = [list(range(self.traffic.num_vehicles)) for _ in range(request.batch_size)]

        for batch in range(request.batch_size):
            types = [self._vehicle_type(request, batch, vehicle) for vehicle in range(self.traffic.num_vehicles)]
            dimensions = [list(_VEHICLES[code][:3]) for code in types]
            antennas = [_VEHICLES[code][3] for code in types]
            initial = self._initial_states(request, batch, types)
            positions, velocities, orientations, active, lane_ids, street_ids = self._evolve(
                request, batch, initial, antennas
            )
            batches_positions.append(positions)
            batches_velocities.append(velocities)
            batches_orientations.append(orientations)
            batches_active.append(active)
            batches_lanes.append(lane_ids)
            batches_streets.append(street_ids)
            batches_types.append(types)
            batches_dimensions.append(dimensions)
            batches_antennas.append(antennas)

        dtype, device = execution.float_dtype, execution.device
        times = [step * self.traffic.update_interval_s for step in range(request.num_mobility_steps)]
        return V2XSceneTrajectory(
            backend.array(batches_positions, dtype=dtype, device=device),
            backend.array(batches_velocities, dtype=dtype, device=device),
            backend.array(batches_orientations, dtype=dtype, device=device),
            backend.array(batches_active, dtype="bool", device=device),
            backend.array(batches_lanes, dtype="int64", device=device),
            backend.array(batches_streets, dtype="int64", device=device),
            backend.array(vehicle_ids, dtype="int64", device=device),
            backend.array(batches_types, dtype="int64", device=device),
            backend.array(batches_dimensions, dtype=dtype, device=device),
            backend.array(batches_antennas, dtype=dtype, device=device),
            backend.array(times, dtype=dtype, device=device),
            self._metadata(request),
        )

    def realize_sequence(
        self,
        channel,
        sampling,
        scene: V2XSceneTrajectory,
        tx_vehicle_indices: tuple[int, ...],
        rx_vehicle_indices: tuple[int, ...],
    ) -> tuple:
        """Realize every mobility snapshot while carrying immutable state."""

        state = None
        results = []
        for snapshot in range(scene.shape[2]):
            request = V2XChannelSamplingRequest(
                sampling, scene, snapshot, tx_vehicle_indices, rx_vehicle_indices, state
            )
            realization = channel.realize(request)
            results.append(realization)
            state = realization.state
        return tuple(results)

    def _initial_states(
        self, request: V2XSceneSamplingRequest, batch: int, types: list[int]
    ) -> list[tuple[float, float, int, bool]]:
        assignments: list[list[int]] = [[] for _ in self.grid.lanes]
        for vehicle in range(self.traffic.num_vehicles):
            assignments[vehicle % len(self.grid.lanes)].append(vehicle)
        result: list[tuple[float, float, int, bool] | None] = [None] * len(types)
        for lane, vehicles in zip(self.grid.lanes, assignments, strict=True):
            if not vehicles:
                continue
            uniforms = request.random_stream.fork(
                f"tr37885.scene.{batch}.lane.{lane.lane_id}.gaps"
            ).uniform((len(vehicles),), 1.0e-9, 1.0, dtype=request.execution.float_dtype)
            longitudinal = 0.0
            speed = self._speed(lane)
            mean_gap = max(speed * 2.0 * self.traffic.density_multiplier, 2.0)
            for ordinal, vehicle in enumerate(vehicles):
                gap = max(2.0, -mean_gap * log(float(request.execution.array_backend.to_scalar(uniforms[ordinal]))))
                longitudinal += gap + _VEHICLES[types[vehicle]][0]
                coordinate = longitudinal if lane.direction > 0 else lane.end_m - longitudinal
                active = lane.start_m <= coordinate <= lane.end_m
                x = coordinate if lane.orientation == "horizontal" else lane.coordinate_m
                y = lane.coordinate_m if lane.orientation == "horizontal" else coordinate
                result[vehicle] = (x, y, lane.lane_id, active)
        return [item if item is not None else (0.0, 0.0, 0, False) for item in result]

    def _evolve(
        self,
        request: V2XSceneSamplingRequest,
        batch: int,
        initial: list[tuple[float, float, int, bool]],
        antenna_heights: list[float],
    ) -> tuple[list, list, list, list, list, list]:
        vehicle_count, steps = self.traffic.num_vehicles, request.num_mobility_steps
        positions = [[] for _ in range(vehicle_count)]
        velocities = [[] for _ in range(vehicle_count)]
        orientations = [[] for _ in range(vehicle_count)]
        active_values = [[] for _ in range(vehicle_count)]
        lane_values = [[] for _ in range(vehicle_count)]
        street_values = [[] for _ in range(vehicle_count)]
        states = list(initial)
        for step in range(steps):
            for vehicle, (x, y, lane_id, active) in enumerate(states):
                lane = self.grid.lane(lane_id)
                speed = self._speed(lane)
                vx = speed * lane.direction if lane.orientation == "horizontal" else 0.0
                vy = speed * lane.direction if lane.orientation == "vertical" else 0.0
                yaw = atan2(vy, vx) if speed > 0.0 else (0.0 if lane.orientation == "horizontal" else pi / 2.0)
                positions[vehicle].append([x, y, antenna_heights[vehicle]])
                velocities[vehicle].append([vx, vy, 0.0])
                orientations[vehicle].append([yaw, 0.0, 0.0])
                active_values[vehicle].append(active)
                lane_values[vehicle].append(lane_id if active else -1)
                street_values[vehicle].append(lane.street_id if active else -1)
            if step + 1 < steps:
                states = [self._advance(request, batch, vehicle, step, state) for vehicle, state in enumerate(states)]
        return positions, velocities, orientations, active_values, lane_values, street_values

    def _advance(
        self,
        request: V2XSceneSamplingRequest,
        batch: int,
        vehicle: int,
        step: int,
        state: tuple[float, float, int, bool],
    ) -> tuple[float, float, int, bool]:
        x, y, lane_id, active = state
        if not active:
            return state
        lane = self.grid.lane(lane_id)
        distance = self._speed(lane) * self.traffic.update_interval_s
        old = x if lane.orientation == "horizontal" else y
        new = old + lane.direction * distance
        crossed = self._crossed_intersection(lane, old, new)
        if crossed is not None and self.traffic.option == "A":
            turn = self._turn(request, batch, vehicle, crossed.intersection_id)
            if turn != 0:
                new_lane = self._turned_lane(lane, crossed, turn)
                if new_lane is not None:
                    intersection_coordinate = (
                        crossed.x_m if lane.orientation == "horizontal" else crossed.y_m
                    )
                    remaining = abs(new - intersection_coordinate)
                    lane = new_lane
                    lane_id = lane.lane_id
                    if lane.orientation == "horizontal":
                        x = crossed.x_m + lane.direction * remaining
                        y = lane.coordinate_m
                    else:
                        x = lane.coordinate_m
                        y = crossed.y_m + lane.direction * remaining
                    return x, y, lane_id, True
        if lane.orientation == "horizontal":
            x = new
        else:
            y = new
        longitudinal = x if lane.orientation == "horizontal" else y
        inside = lane.start_m <= longitudinal <= lane.end_m
        if inside:
            return x, y, lane_id, True
        policy = self.grid.config.boundary_policy
        if policy == "exit":
            return x, y, lane_id, False
        if policy in {"periodic", "respawn"}:
            if lane.orientation == "horizontal":
                return x % self.grid.config.width_m, lane.coordinate_m, lane_id, True
            return lane.coordinate_m, y % self.grid.config.height_m, lane_id, True
        raise RuntimeError("unreachable boundary policy")

    def _crossed_intersection(self, lane: V2XLane, old: float, new: float):
        candidates = []
        for intersection in self.grid.intersections:
            coordinate = intersection.x_m if lane.orientation == "horizontal" else intersection.y_m
            if (old < coordinate <= new) or (new <= coordinate < old):
                candidates.append((abs(coordinate - old), intersection))
        return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]

    def _turn(self, request: V2XSceneSamplingRequest, batch: int, vehicle: int, intersection: int) -> int:
        value = float(request.execution.array_backend.to_scalar(
            request.random_stream.fork(
                f"tr37885.vehicle.{batch}.{vehicle}.intersection.{intersection}.turn"
            ).uniform((1,), dtype=request.execution.float_dtype)[0]
        ))
        straight, left, _ = self.traffic.turn_probabilities
        return 0 if value < straight else 1 if value < straight + left else -1

    def _turned_lane(self, lane: V2XLane, intersection, turn: int) -> V2XLane | None:
        if lane.orientation == "horizontal":
            new_direction = lane.direction * turn
            street_id = intersection.vertical_street_id
        else:
            new_direction = -lane.direction * turn
            street_id = intersection.horizontal_street_id
        return next(
            (candidate for candidate in self.grid.lanes
             if candidate.street_id == street_id and candidate.direction == new_direction
             and candidate.speed_slot == lane.speed_slot),
            None,
        )


__all__ = ["TR37885UrbanGridSimulator"]
