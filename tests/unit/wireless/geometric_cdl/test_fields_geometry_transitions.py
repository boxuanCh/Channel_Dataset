from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.geometric_cdl import SpatialFieldBank
from channel_dataset.channels.wireless.geometric_cdl.geometry import (
    NLOS_COMPONENT,
    construct_scatterers,
    drift_paths,
)
from channel_dataset.channels.wireless.geometric_cdl.transitions import segment_amplitude_weights


def test_spatial_field_is_query_order_independent(execution_context):
    backend = execution_context.array_backend
    bank = SpatialFieldBank.sample(
        3, 32, 10.0, execution_context.random_stream(11),
        dtype=execution_context.float_dtype,
    )
    positions = backend.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [4.0, -1.0, 3.0]],
        dtype=execution_context.float_dtype,
    )
    complete = bank.evaluate(positions)
    pieces = backend.stack(tuple(bank.evaluate(positions[index]) for index in range(3)))
    error = backend.max(backend.abs(complete - pieces))
    assert float(backend.to_scalar(error)) < 1.0e-12


def test_transition_amplitudes_conserve_power(execution_context):
    backend = execution_context.array_backend
    weights = segment_amplitude_weights(
        (4,), 8, 0.5, backend, dtype=execution_context.float_dtype,
        device=execution_context.device,
    )
    power = backend.sum(weights * weights, axis=0)
    error = backend.max(backend.abs(power - 1.0))
    assert float(backend.to_scalar(error)) < 1.0e-12
    active = backend.sum(backend.astype(weights > 0.0, "int64"), axis=0)
    assert int(backend.to_scalar(backend.max(active))) <= 2


def test_single_bounce_fallback_reconstructs_length(execution_context):
    backend = execution_context.array_backend
    dtype = execution_context.float_dtype
    tx = backend.array([[[[[[0.0, 0.0, 0.0]]]]]], dtype=dtype)
    rx = backend.array([[[[[[10.0, 0.0, 0.0]]]]]], dtype=dtype)
    angle = backend.zeros((1, 1, 1, 1, 1), dtype=dtype)
    zenith = angle + 0.5 * 3.141592653589793
    delay = angle + 20.0 / 299792458.0
    component = backend.full((1, 1, 1, 1, 1), NLOS_COMPONENT, dtype="int64")
    fbs, lbs, _, _ = construct_scatterers(
        tx, rx, angle, zenith, angle, zenith, delay, component,
        tolerance_m=1.0e-12, max_iterations=1, fallback="single_bounce",
    )
    path, *_ = drift_paths(
        backend.reshape(tx, (1, 1, 1, 3)),
        backend.reshape(rx, (1, 1, 1, 3)), fbs, lbs,
        backend.reshape(component, (1, 1, 1, 1)),
    )
    assert float(backend.to_scalar(path[0, 0, 0, 0, 0, 0])) == pytest.approx(30.0)
