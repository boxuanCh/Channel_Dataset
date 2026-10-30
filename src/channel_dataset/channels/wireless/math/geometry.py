"""Wavelength, Doppler, and Cartesian link-geometry helpers."""

from __future__ import annotations

from math import isfinite
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.core import ExecutionContext, Vector3

from ._utils import as_real_array
from .constants import SPEED_OF_LIGHT_MPS


def wavelength_m(carrier_frequency_hz: Any) -> Any:
    """Return free-space wavelength for a positive carrier frequency.

    Parameters
    ----------
    carrier_frequency_hz : float or backend array
        Positive frequency in hertz.

    Returns
    -------
    float or backend array
        Wavelength in metres.
    """

    try:
        backend = get_backend(backend_name_for_array(carrier_frequency_hz))
    except TypeError:
        frequency = float(carrier_frequency_hz)
        if not isfinite(frequency) or frequency <= 0.0:
            raise ValueError("carrier_frequency_hz must be finite and positive")
        return SPEED_OF_LIGHT_MPS / frequency
    if (
        not backend.all_finite(carrier_frequency_hz)
        or backend.to_scalar(backend.min(carrier_frequency_hz)) <= 0.0
    ):
        raise ValueError("carrier_frequency_hz must contain finite positive values")
    return SPEED_OF_LIGHT_MPS / carrier_frequency_hz


def doppler_shift_hz(carrier_frequency_hz: Any, closing_speed_mps: Any) -> Any:
    """Compute Doppler shift from carrier frequency and closing speed.

    Positive ``closing_speed_mps`` means the endpoints approach one another
    and therefore produces a positive frequency shift.
    """

    return closing_speed_mps / wavelength_m(carrier_frequency_hz)


def _vector_array(value: Any, execution: ExecutionContext, name: str) -> Any:
    if isinstance(value, Vector3):
        value = (value.x, value.y, value.z)
    array = as_real_array(value, execution, name)
    shape = execution.array_backend.shape(array)
    if not shape or shape[-1] != 3:
        raise ValueError(f"{name} must have final dimension 3")
    return array


def euclidean_distance(
    first_position_m: Any,
    second_position_m: Any,
    execution: ExecutionContext,
) -> Any:
    """Compute Euclidean distance between Cartesian positions.

    Inputs may be :class:`~channel_dataset.core.Vector3` objects or broadcastable
    arrays ending in the ``xyz`` dimension. The result uses metres.
    """

    backend = execution.array_backend
    first = _vector_array(first_position_m, execution, "first_position_m")
    second = _vector_array(second_position_m, execution, "second_position_m")
    difference = second - first
    return backend.sqrt(backend.sum(difference * difference, axis=-1))


def line_of_sight_unit_vector(
    transmitter_position_m: Any,
    receiver_position_m: Any,
    execution: ExecutionContext,
) -> Any:
    """Return unit vectors pointing from transmitter to receiver.

    Raises
    ------
    ValueError
        If any transmitter and receiver positions are collocated.
    """

    backend = execution.array_backend
    transmitter = _vector_array(transmitter_position_m, execution, "transmitter_position_m")
    receiver = _vector_array(receiver_position_m, execution, "receiver_position_m")
    difference = receiver - transmitter
    distance = backend.sqrt(backend.sum(difference * difference, axis=-1))
    if backend.to_scalar(backend.min(distance)) <= 0.0:
        raise ValueError("collocated endpoints do not define a line of sight")
    return difference / backend.reshape(distance, backend.shape(distance) + (1,))


def closing_speed_mps(
    transmitter_position_m: Any,
    receiver_position_m: Any,
    transmitter_velocity_mps: Any,
    receiver_velocity_mps: Any,
    execution: ExecutionContext,
) -> Any:
    """Return relative speed along the line of sight, positive when closing.

    Tangential velocity contributes zero. A positive result means link distance
    is decreasing and can be passed directly to :func:`doppler_shift_hz`.
    """

    backend = execution.array_backend
    direction = line_of_sight_unit_vector(
        transmitter_position_m, receiver_position_m, execution
    )
    transmitter_velocity = _vector_array(
        transmitter_velocity_mps, execution, "transmitter_velocity_mps"
    )
    receiver_velocity = _vector_array(
        receiver_velocity_mps, execution, "receiver_velocity_mps"
    )
    separation_velocity = receiver_velocity - transmitter_velocity
    return -backend.sum(separation_velocity * direction, axis=-1)


__all__ = [
    "closing_speed_mps",
    "doppler_shift_hz",
    "euclidean_distance",
    "line_of_sight_unit_vector",
    "wavelength_m",
]
