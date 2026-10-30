from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.fading import (
    FlatMIMOFading,
    FlatMIMOFadingConfig,
    apply_kronecker_correlation,
    apply_per_column_receive_correlation,
)


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def _base_config() -> FlatMIMOFadingConfig:
    return FlatMIMOFadingConfig(
        num_rx=1,
        num_rx_antennas=2,
        num_tx=1,
        num_tx_antennas=2,
        average_power_linear=1.0,
        power_normalization="per_coefficient",
        correlation_mode="none",
        rx_correlation_coefficient=None,
        tx_correlation_coefficient=None,
        per_column_rx_correlation_coefficients=None,
    )


def _request(execution_context, batch_size: int) -> ChannelSamplingRequest:
    return ChannelSamplingRequest(
        batch_size=batch_size,
        num_time_samples=1,
        sampling_frequency_hz=1.0,
        execution=execution_context,
        random_stream=execution_context.random_stream(1234),
    )


def test_identity_kronecker_correlation_leaves_coefficients_unchanged(backend) -> None:
    values = backend.array(
        [[[1.0 + 2.0j, 3.0 - 1.0j], [2.0 + 0.5j, -1.0j]]], dtype="complex128"
    )
    identity = backend.array(np.eye(2), dtype="float64")
    result = apply_kronecker_correlation(values, identity, identity)
    assert np.allclose(_as_numpy(result), _as_numpy(values))


def test_identity_per_column_correlation_leaves_coefficients_unchanged(backend) -> None:
    values = backend.array(np.arange(8).reshape(2, 2, 2) + 1j, dtype="complex128")
    identities = backend.array(np.stack([np.eye(2), np.eye(2)]), dtype="float64")
    result = apply_per_column_receive_correlation(values, identities)
    assert np.allclose(_as_numpy(result), _as_numpy(values))


def test_kronecker_empirical_receive_covariance(execution_context) -> None:
    config = replace(
        _base_config(),
        num_tx_antennas=1,
        correlation_mode="kronecker",
        rx_correlation_coefficient=0.75,
        tx_correlation_coefficient=0.0,
    )
    response = FlatMIMOFading(config).realize(_request(execution_context, 40_000))
    samples = _as_numpy(response.coefficients)[:, 0, :, 0, 0]
    covariance = samples.T @ samples.conj() / samples.shape[0]
    assert np.allclose(covariance, [[1.0, 0.75], [0.75, 1.0]], atol=0.025)


def test_per_column_receive_covariance(execution_context) -> None:
    config = replace(
        _base_config(),
        correlation_mode="per_column_receive",
        per_column_rx_correlation_coefficients=(0.0, 0.8),
    )
    response = FlatMIMOFading(config).realize(_request(execution_context, 40_000))
    samples = _as_numpy(response.coefficients)[:, 0, :, 0, :]
    first_covariance = samples[:, :, 0].T @ samples[:, :, 0].conj() / samples.shape[0]
    second_covariance = samples[:, :, 1].T @ samples[:, :, 1].conj() / samples.shape[0]
    assert first_covariance[0, 1] == pytest.approx(0.0, abs=0.025)
    assert second_covariance[0, 1].real == pytest.approx(0.8, abs=0.025)


def test_correlation_shape_mismatch_is_rejected(backend) -> None:
    values = backend.ones((3, 2, 2), dtype="complex128")
    wrong = backend.array(np.eye(3), dtype="float64")
    identity = backend.array(np.eye(2), dtype="float64")
    with pytest.raises(ValueError, match="receive_correlation shape"):
        apply_kronecker_correlation(values, wrong, identity)
