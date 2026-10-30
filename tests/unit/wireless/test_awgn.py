from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.noise import (
    AdditiveWhiteGaussianNoise,
    AWGNConfig,
    parse_awgn_config,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_awgn_complex_variance_mean_and_circular_symmetry(execution_context) -> None:
    source = AdditiveWhiteGaussianNoise(AWGNConfig(2.0))
    samples = _as_numpy(
        source.sample(
            (200_000,),
            execution_context.random_stream(123),
            dtype=execution_context.complex_dtype,
        )
    )
    assert abs(np.mean(samples)) < 0.015
    assert np.mean(np.abs(samples) ** 2) == pytest.approx(2.0, rel=0.02)
    assert np.var(samples.real) == pytest.approx(1.0, rel=0.03)
    assert np.var(samples.imag) == pytest.approx(1.0, rel=0.03)
    assert abs(np.mean(samples**2)) < 0.02


def test_awgn_seeded_replay(execution_context) -> None:
    source = AdditiveWhiteGaussianNoise(AWGNConfig(0.25))
    first = source.sample(
        (16, 8), execution_context.random_stream(77), dtype=execution_context.complex_dtype
    )
    second = source.sample(
        (16, 8), execution_context.random_stream(77), dtype=execution_context.complex_dtype
    )
    assert np.array_equal(_as_numpy(first), _as_numpy(second))


def test_awgn_apply_preserves_shape_dtype_and_backend(execution_context) -> None:
    backend = execution_context.array_backend
    signal = backend.zeros(
        (3, 4), dtype=execution_context.complex_dtype, device=execution_context.device
    )
    source = AdditiveWhiteGaussianNoise(AWGNConfig(0.5))
    output = source.apply(signal, execution_context.random_stream(9))
    assert backend.shape(output) == (3, 4)
    assert backend.dtype(output) == backend.dtype(signal)
    assert np.any(_as_numpy(output) != 0.0)


def test_zero_variance_awgn_is_exactly_zero(execution_context) -> None:
    source = AdditiveWhiteGaussianNoise(AWGNConfig(0.0))
    samples = source.sample(
        (4, 4), execution_context.random_stream(1), dtype=execution_context.complex_dtype
    )
    assert np.array_equal(_as_numpy(samples), np.zeros((4, 4)))


def test_awgn_config_is_strict() -> None:
    assert parse_awgn_config({"noise_variance_linear": 0.2}).noise_variance_linear == 0.2
    with pytest.raises(ValueError, match="unknown AWGN"):
        parse_awgn_config({"noise_variance_linear": 1.0, "variance_db": 0.0})
    with pytest.raises(ValueError, match="nonnegative"):
        AWGNConfig(-1.0)


def test_awgn_rejects_real_signal(execution_context) -> None:
    signal = execution_context.array_backend.ones(
        (2,), dtype=execution_context.float_dtype, device=execution_context.device
    )
    with pytest.raises(TypeError, match="complex"):
        AdditiveWhiteGaussianNoise(AWGNConfig(1.0)).apply(
            signal, execution_context.random_stream(1)
        )
