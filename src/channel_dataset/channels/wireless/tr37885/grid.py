"""Reference urban-grid geometry from TR 37.885 Annex A."""

from __future__ import annotations

from dataclasses import dataclass

from channel_dataset.channels.wireless.tr37885.config import TR37885UrbanGridConfig


@dataclass(frozen=True, slots=True)
class V2XStreet:
    """One horizontal or vertical street centerline."""

    street_id: int
    orientation: str
    coordinate_m: float
    start_m: float
    end_m: float


@dataclass(frozen=True, slots=True)
class V2XLane:
    """One directed lane centerline."""

    lane_id: int
    street_id: int
    orientation: str
    direction: int
    coordinate_m: float
    start_m: float
    end_m: float
    speed_slot: int


@dataclass(frozen=True, slots=True)
class V2XIntersection:
    """Intersection between a horizontal and vertical street."""

    intersection_id: int
    x_m: float
    y_m: float
    horizontal_street_id: int
    vertical_street_id: int


@dataclass(frozen=True, slots=True)
class V2XBuildingBlock:
    """Axis-aligned building-obstruction footprint."""

    block_id: int
    minimum_x_m: float
    maximum_x_m: float
    minimum_y_m: float
    maximum_y_m: float


class TR37885UrbanGrid:
    """Generate the exact rectangular Annex-A reference road geometry."""

    def __init__(self, config: TR37885UrbanGridConfig | None = None) -> None:
        self.config = config or TR37885UrbanGridConfig()
        self.streets = self._streets()
        self.lanes = self._lanes()
        self.intersections = self._intersections()
        self.building_blocks = self._blocks()

    def _streets(self) -> tuple[V2XStreet, ...]:
        result: list[V2XStreet] = []
        identifier = 0
        for index in range(self.config.cells_y + 1):
            result.append(V2XStreet(identifier, "horizontal", index * self.config.cell_height_m, 0.0, self.config.width_m))
            identifier += 1
        for index in range(self.config.cells_x + 1):
            result.append(V2XStreet(identifier, "vertical", index * self.config.cell_width_m, 0.0, self.config.height_m))
            identifier += 1
        return tuple(result)

    def _lanes(self) -> tuple[V2XLane, ...]:
        lanes: list[V2XLane] = []
        identifier = 0
        n = self.config.lanes_per_direction
        width = self.config.lane_width_m
        for street in self.streets:
            for direction in (-1, 1):
                for slot in range(n):
                    offset = direction * (slot + 0.5) * width
                    lanes.append(V2XLane(identifier, street.street_id, street.orientation, direction, street.coordinate_m + offset, street.start_m, street.end_m, slot))
                    identifier += 1
        return tuple(lanes)

    def _intersections(self) -> tuple[V2XIntersection, ...]:
        horizontals = [street for street in self.streets if street.orientation == "horizontal"]
        verticals = [street for street in self.streets if street.orientation == "vertical"]
        result = []
        for horizontal in horizontals:
            for vertical in verticals:
                result.append(V2XIntersection(len(result), vertical.coordinate_m, horizontal.coordinate_m, horizontal.street_id, vertical.street_id))
        return tuple(result)

    def _blocks(self) -> tuple[V2XBuildingBlock, ...]:
        half = self.config.street_width_m / 2.0
        result = []
        for x in range(self.config.cells_x):
            for y in range(self.config.cells_y):
                result.append(V2XBuildingBlock(
                    len(result), x * self.config.cell_width_m + half,
                    (x + 1) * self.config.cell_width_m - half,
                    y * self.config.cell_height_m + half,
                    (y + 1) * self.config.cell_height_m - half,
                ))
        return tuple(result)

    def lane(self, lane_id: int) -> V2XLane:
        """Return a lane by stable identifier."""

        if not 0 <= lane_id < len(self.lanes):
            raise IndexError("lane_id is outside the grid")
        return self.lanes[lane_id]

    def segment_intersects_building(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Return whether a link segment crosses any building-block interior."""

        return any(_segment_rectangle(start, end, block) for block in self.building_blocks)


def _segment_rectangle(start: tuple[float, float], end: tuple[float, float], block: V2XBuildingBlock) -> bool:
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    lower, upper = 0.0, 1.0
    for p, q in (
        (-dx, x0 - block.minimum_x_m), (dx, block.maximum_x_m - x0),
        (-dy, y0 - block.minimum_y_m), (dy, block.maximum_y_m - y0),
    ):
        if p == 0.0:
            if q <= 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower >= upper:
            return False
    return 0.0 < upper and lower < 1.0


__all__ = [
    "TR37885UrbanGrid", "V2XBuildingBlock", "V2XIntersection", "V2XLane", "V2XStreet",
]
