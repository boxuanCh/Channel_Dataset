"""TR 37.885 6 GHz vehicle-antenna Option-1 profiles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi

from channel_dataset.channels.wireless.tr37885.profiles import load_urban_parameters


@dataclass(frozen=True, slots=True)
class V2XVehicleAntennaProfile:
    """Validated Option-1 vehicle antenna placement.

    Parameters
    ----------
    vehicle_type_code : {1, 2, 3}
        Standard vehicle type.
    locations : tuple of str
        Physical antenna locations, such as front and rear.
    bearing_rad : tuple of float
        Location bearing relative to vehicle yaw.
    maximum_element_gain_db : float
        Maximum 6 GHz element gain.
    element_spacing_wavelengths : float
        Standard inter-element spacing normalized by wavelength.
    tilt_rad : float
        Array tilt in radians.
    baseline_array : tuple of tuple of int
        (M, N, P, Mg, Ng) baseline layout for each location.
    optional_array : tuple of tuple of int
        Optional layout for each location.
    """

    vehicle_type_code: int
    locations: tuple[str, ...]
    bearing_rad: tuple[float, ...]
    maximum_element_gain_db: float
    element_spacing_wavelengths: float
    tilt_rad: float
    baseline_array: tuple[tuple[int, int, int, int, int], ...]
    optional_array: tuple[tuple[int, int, int, int, int], ...]

    def __post_init__(self) -> None:
        count = len(self.locations)
        if self.vehicle_type_code not in {1, 2, 3}:
            raise ValueError("vehicle_type_code must be 1, 2, or 3")
        if count == 0 or len(self.bearing_rad) != count:
            raise ValueError("vehicle antenna locations and bearings must align")
        if len(self.baseline_array) != count or len(self.optional_array) != count:
            raise ValueError("vehicle antenna layouts must align with locations")
        if self.element_spacing_wavelengths <= 0.0:
            raise ValueError("element_spacing_wavelengths must be positive")

    def port_bearings_rad(self, num_ports: int, vehicle_yaw_rad: float) -> tuple[float, ...]:
        """Return yaw-rotated location bearings for a requested port count.

        Ports are distributed deterministically across the standard physical
        locations. A SISO request uses the first location; multiport requests
        cycle through all locations without changing random streams.
        """

        if isinstance(num_ports, bool) or not isinstance(num_ports, int):
            raise TypeError("num_ports must be an integer")
        if num_ports <= 0:
            raise ValueError("num_ports must be positive")
        return tuple(
            vehicle_yaw_rad + self.bearing_rad[index % len(self.bearing_rad)]
            for index in range(num_ports)
        )


@lru_cache(maxsize=3)
def load_vehicle_antenna_profile(vehicle_type_code: int) -> V2XVehicleAntennaProfile:
    """Load the 6 GHz Option-1 antenna placement for one vehicle type.

    Parameters
    ----------
    vehicle_type_code : {1, 2, 3}
        Standard vehicle type identifier.

    Returns
    -------
    V2XVehicleAntennaProfile
        Immutable placement, bearing, gain, spacing, and layout information.
    """

    if vehicle_type_code not in {1, 2, 3}:
        raise ValueError("vehicle_type_code must be 1, 2, or 3")
    antenna = load_urban_parameters()["vehicle_antenna"]
    values = antenna["types"][vehicle_type_code]
    locations = tuple(str(value) for value in values["locations"])
    bearings = tuple(float(value) * pi / 180.0 for value in values["bearing_degrees"])
    if vehicle_type_code == 2:
        baseline = (tuple(int(value) for value in values["baseline_array"]),)
        optional = (tuple(int(value) for value in values["optional_array"]),)
    else:
        baseline_value = tuple(
            int(value) for value in values["baseline_array_per_location"]
        )
        optional_value = tuple(
            int(value) for value in values["optional_array_per_location"]
        )
        baseline = tuple(baseline_value for _ in locations)
        optional = tuple(optional_value for _ in locations)
    return V2XVehicleAntennaProfile(
        vehicle_type_code=vehicle_type_code,
        locations=locations,
        bearing_rad=bearings,
        maximum_element_gain_db=float(antenna["maximum_element_gain_db"]),
        element_spacing_wavelengths=float(antenna["element_spacing_wavelengths"]),
        tilt_rad=float(antenna["tilt_degrees"]) * pi / 180.0,
        baseline_array=baseline,
        optional_array=optional,
    )


__all__ = ["V2XVehicleAntennaProfile", "load_vehicle_antenna_profile"]
