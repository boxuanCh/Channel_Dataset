from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    doppler_phase_evolution,
)
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS


def test_doppler_matches_projected_velocity(backend, execution_context):
    direction = backend.array([[1.0, 0.0, 0.0]], dtype=execution_context.float_dtype)
    rx_velocity = backend.array([3.0, 0.0, 0.0], dtype=execution_context.float_dtype)
    tx_velocity = backend.zeros((3,), dtype=execution_context.float_dtype)
    request = ChannelSamplingRequest(
        1, 2, 1000.0, execution_context, execution_context.random_stream(1)
    )
    doppler, phase = doppler_phase_evolution(
        direction, tx_velocity[None, :] * 0.0 + direction, rx_velocity,
        tx_velocity, 3.5e9, request
    )
    assert abs(float(backend.to_scalar(doppler[0])) - 3.0 * 3.5e9 / SPEED_OF_LIGHT_MPS) < 1e-9
    assert backend.shape(phase) == (1, 2)
