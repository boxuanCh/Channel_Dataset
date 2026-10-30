"""Shared numerical primitives for wireless channel models."""

from channel_dataset.channels.wireless.math.angles import (
    degrees_to_radians,
    radians_to_degrees,
    wrap_angle_degrees,
    wrap_angle_radians,
)
from channel_dataset.channels.wireless.math.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    SPEED_OF_LIGHT_MPS,
)
from channel_dataset.channels.wireless.math.conversions import (
    amplitude_db_to_linear,
    linear_amplitude_to_db,
    linear_power_to_db,
    power_db_to_linear,
)
from channel_dataset.channels.wireless.math.correlation import (
    exponential_correlation,
    one_ring_correlation,
)
from channel_dataset.channels.wireless.math.directions import (
    cartesian_to_spherical,
    spherical_to_cartesian,
)
from channel_dataset.channels.wireless.math.geometry import (
    closing_speed_mps,
    doppler_shift_hz,
    euclidean_distance,
    line_of_sight_unit_vector,
    wavelength_m,
)
from channel_dataset.channels.wireless.math.linalg import matrix_sqrt_psd
from channel_dataset.channels.wireless.math.random import complex_normal

__all__ = [
    "BOLTZMANN_CONSTANT_J_PER_K",
    "SPEED_OF_LIGHT_MPS",
    "amplitude_db_to_linear",
    "cartesian_to_spherical",
    "closing_speed_mps",
    "complex_normal",
    "degrees_to_radians",
    "doppler_shift_hz",
    "euclidean_distance",
    "exponential_correlation",
    "line_of_sight_unit_vector",
    "linear_amplitude_to_db",
    "linear_power_to_db",
    "matrix_sqrt_psd",
    "one_ring_correlation",
    "power_db_to_linear",
    "radians_to_degrees",
    "spherical_to_cartesian",
    "wavelength_m",
    "wrap_angle_degrees",
    "wrap_angle_radians",
]
