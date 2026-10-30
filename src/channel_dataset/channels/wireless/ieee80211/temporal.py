"""Fixed-window IEEE indoor temporal-fading processes."""

from __future__ import annotations

from math import pi, sqrt
from typing import Any

from channel_dataset.backends import ArrayBackend, RandomStream
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS


def bell_doppler_process(
    stream: RandomStream[Any],
    backend: ArrayBackend[Any],
    shape: tuple[int, ...],
    num_time_samples: int,
    sampling_frequency_hz: float,
    carrier_frequency_hz: float,
    environmental_speed_mps: float,
    *,
    num_sinusoids: int,
    float_dtype: str,
    complex_dtype: str,
    device: str,
) -> Any:
    r"""Sample the TGn Bell-spectrum complex Gaussian process.

    The two-sided spectral shape is
    :math:`S(f) \propto 1/(1+9(f/f_d)^2)` truncated at ``5*f_d``, where
    :math:`f_d=v/\lambda` (IEEE 802.11-03/940r4, section 4.7.1).
    At zero speed the returned coefficient is constant over time.
    """

    if num_sinusoids <= 0:
        raise ValueError("num_sinusoids must be positive")
    maximum_doppler_hz = environmental_speed_mps * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    if maximum_doppler_hz > 0 and sampling_frequency_hz <= 10.0 * maximum_doppler_hz:
        raise ValueError("sampling_frequency_hz must exceed the truncated Bell Doppler bandwidth")
    if maximum_doppler_hz == 0:
        initial = stream.complex_normal(shape, variance=1.0, dtype=complex_dtype)
        return backend.broadcast_to(initial[..., None], (*shape, num_time_samples))
    frequencies = [
        -5.0 * maximum_doppler_hz
        + 10.0 * maximum_doppler_hz * (index + 0.5) / num_sinusoids
        for index in range(num_sinusoids)
    ]
    powers = [1.0 / (1.0 + 9.0 * (frequency / maximum_doppler_hz) ** 2) for frequency in frequencies]
    total = sum(powers)
    amplitude = backend.array(
        [sqrt(value / total) for value in powers], dtype=float_dtype, device=device
    )
    frequency = backend.array(frequencies, dtype=float_dtype, device=device)
    time = backend.arange(num_time_samples, dtype=float_dtype, device=device) / sampling_frequency_hz
    oscillators = stream.complex_normal((*shape, num_sinusoids), variance=1.0, dtype=complex_dtype)
    phase = backend.exp(1j * 2.0 * pi * frequency[:, None] * time[None, :])
    return backend.einsum("...k,k,kt->...t", oscillators, amplitude, phase)


def apply_fluorescent_effect(
    values: Any,
    stream: RandomStream[Any],
    backend: ArrayBackend[Any],
    sampling_frequency_hz: float,
    *,
    float_dtype: str,
    device: str,
) -> Any:
    """Apply the TGn 100/300/500-Hz fluorescent modulation approximation."""

    if sampling_frequency_hz <= 1000.0:
        raise ValueError("fluorescent_effect requires sampling_frequency_hz above 1 kHz")
    samples = backend.shape(values)[-1]
    time = backend.arange(samples, dtype=float_dtype, device=device) / sampling_frequency_hz
    interference_ratio = backend.abs(stream.normal((1,), 0.0203, 0.0107, dtype=float_dtype))[0]
    phases = stream.uniform((3,), 0.0, 2.0 * pi, dtype=float_dtype)
    amplitudes = backend.array((1.0, 10.0 ** (-15.0 / 20.0), 0.1), dtype=float_dtype, device=device)
    frequencies = backend.array((100.0, 300.0, 500.0), dtype=float_dtype, device=device)
    modulation = backend.sum(
        amplitudes[:, None] * backend.exp(1j * (2.0 * pi * frequencies[:, None] * time + phases[:, None])),
        axis=0,
    )
    # Section 4.7.3 draws X~N(0.0203, 0.0107^2), with I/C=X^2.
    modulation = modulation * (
        interference_ratio
        / sqrt(sum(x * x for x in (1.0, 10.0 ** (-15.0 / 20.0), 0.1)))
    )
    return values * (1.0 + modulation)


def apply_moving_vehicle_effect(
    values: Any,
    stream: RandomStream[Any],
    backend: ArrayBackend[Any],
    sampling_frequency_hz: float,
    carrier_frequency_hz: float,
    *,
    float_dtype: str,
    device: str,
) -> Any:
    """Add the profile-F 40-km/h positive-frequency Doppler component."""

    spike_hz = (40.0 / 3.6) * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    if sampling_frequency_hz <= 2.1 * spike_hz:
        raise ValueError("moving_vehicle_effect sampling rate aliases the Doppler spike")
    samples = backend.shape(values)[-1]
    time = backend.arange(samples, dtype=float_dtype, device=device) / sampling_frequency_hz
    component_count = 16
    relative_offsets = [
        -0.05 + 0.1 * (index + 0.5) / component_count for index in range(component_count)
    ]
    frequencies = backend.array(
        [spike_hz * (1.0 + value) for value in relative_offsets],
        dtype=float_dtype,
        device=device,
    )
    spectral_power = [1.0 / (1.0 + 90000.0 * value * value) for value in relative_offsets]
    total_power = sum(spectral_power)
    amplitudes = backend.array(
        [sqrt(value / total_power) for value in spectral_power],
        dtype=float_dtype,
        device=device,
    )
    weights = stream.complex_normal(
        (component_count,), variance=1.0, dtype=backend.dtype(values)
    )
    phase = backend.exp(1j * 2.0 * pi * frequencies[:, None] * time[None, :])
    spike = backend.einsum("k,k,kt->t", weights, amplitudes, phase)
    return (values + sqrt(0.5) * spike) / sqrt(1.5)


__all__ = ["apply_fluorescent_effect", "apply_moving_vehicle_effect", "bell_doppler_process"]
