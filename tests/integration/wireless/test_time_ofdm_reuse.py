from __future__ import annotations

import numpy as np

from channel_dataset.channels.wireless.fading import (
    RayleighBlockFading,
    RayleighBlockFadingConfig,
)
from channel_dataset.channels.wireless.operators import apply_ofdm, apply_time
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.transforms import cir_to_ofdm, cir_to_time
from channel_dataset.core import ExecutionContext


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_one_cir_is_reused_in_time_and_ofdm_domains(backend, backend_name) -> None:
    execution = ExecutionContext(backend_name, "double")
    request = ChannelSamplingRequest(
        batch_size=1,
        num_time_samples=4,
        sampling_frequency_hz=1.0,
        execution=execution,
        random_stream=execution.random_stream(2026),
    )
    model = RayleighBlockFading(
        RayleighBlockFadingConfig(
            num_rx=1,
            num_rx_antennas=1,
            num_tx=1,
            num_tx_antennas=1,
            average_power_linear=1.0,
        )
    )
    cir = model.realize(request)

    time_response = cir_to_time(cir, 1.0, lag_min=0, lag_max=0)
    waveform = backend.ones((1, 1, 1, 4), dtype="complex128")
    first_time = apply_time(time_response, waveform)
    second_time = apply_time(time_response, waveform)

    frequencies = backend.array([-1.0, 0.0, 1.0], dtype="float64")
    ofdm_response = cir_to_ofdm(cir, frequencies)
    grid = backend.ones((1, 1, 1, 4, 3), dtype="complex128")
    first_ofdm = apply_ofdm(ofdm_response, grid)
    second_ofdm = apply_ofdm(ofdm_response, grid)

    assert np.array_equal(_as_numpy(first_time), _as_numpy(second_time))
    assert np.array_equal(_as_numpy(first_ofdm), _as_numpy(second_ofdm))
    assert np.allclose(_as_numpy(first_ofdm), _as_numpy(first_time)[..., None])
