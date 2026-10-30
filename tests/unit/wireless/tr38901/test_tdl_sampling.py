from __future__ import annotations

import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import TappedDelayLine, TDLConfig


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _model(**overrides):
    values = {
        "profile": "TDL-A", "standard_version": "16.1.0",
        "delay_spread_s": 100e-9, "carrier_frequency_hz": 3.5e9,
        "bandwidth_hz": 100e6, "speed_mps": 3.0, "num_rx": 2,
        "num_rx_antennas": 2, "num_tx": 1, "num_tx_antennas": 2,
        "num_sinusoids": 16, "k_factor_db": None, "correlation_mode": "none",
        "rx_correlation_coefficient": None, "tx_correlation_coefficient": None,
        "joint_correlation_matrix": None,
    }
    values.update(overrides)
    return TappedDelayLine(TDLConfig(**values))


def _request(execution_context, seed=77, batch_size=3, time=5):
    return ChannelSamplingRequest(
        batch_size=batch_size, num_time_samples=time, sampling_frequency_hz=2000.0,
        execution=execution_context, random_stream=execution_context.random_stream(seed),
        request_id="tdl-test",
    )


def test_tdl_sampling_shape_dtype_and_provenance(execution_context) -> None:
    realization = _model().realize(_request(execution_context))
    assert realization.shape == (3, 2, 2, 1, 2, 23, 5)
    assert realization.delays_s.shape == (3, 2, 1, 23)
    assert str(realization.coefficients.dtype).endswith("complex128")
    assert realization.metadata.model_name == "tdl"
    assert realization.metadata.request_id == "tdl-test"


def test_tdl_seeded_replay(execution_context) -> None:
    model = _model()
    first = model.realize(_request(execution_context, 123))
    second = model.realize(_request(execution_context, 123))
    assert np.array_equal(_as_numpy(first.coefficients), _as_numpy(second.coefficients))
    assert np.array_equal(_as_numpy(first.delays_s), _as_numpy(second.delays_s))


def test_tdl_batches_are_independent(execution_context) -> None:
    realization = _model().realize(_request(execution_context, batch_size=2))
    values = _as_numpy(realization.coefficients)
    assert not np.array_equal(values[0], values[1])


def test_mask_is_not_needed_for_fixed_tap_count(execution_context) -> None:
    realization = _model().realize(_request(execution_context))
    assert realization.path_mask is None
