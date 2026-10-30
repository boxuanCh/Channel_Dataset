from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.fading import (
    FlatMIMOFading,
    FlatMIMOFadingConfig,
    RayleighBlockFading,
    RayleighBlockFadingConfig,
)
from channel_dataset.channels.wireless.noise import AdditiveWhiteGaussianNoise, AWGNConfig
from tests.statistical.statistical_assertions import (
    as_numpy,
    assert_exponential_power,
    assert_unit_circularity,
)

pytestmark = pytest.mark.statistical


def test_awgn_has_complex_gaussian_distribution(execution_context) -> None:
    samples = as_numpy(
        AdditiveWhiteGaussianNoise(AWGNConfig(1.7)).sample(
            (80_000,),
            execution_context.random_stream(70_001),
            dtype=execution_context.complex_dtype,
        )
    )
    assert_exponential_power(samples, 1.7)
    assert_unit_circularity(samples)


def test_rayleigh_block_fading_has_complex_gaussian_distribution(execution_context) -> None:
    request = ChannelSamplingRequest(
        50_000, 1, 1.0, execution_context, execution_context.random_stream(70_002)
    )
    model = RayleighBlockFading(RayleighBlockFadingConfig(1, 1, 1, 1, 2.5))
    samples = as_numpy(model.realize(request).coefficients).reshape(-1)
    assert_exponential_power(samples, 2.5)
    assert_unit_circularity(samples)


@pytest.mark.parametrize(
    ("mode", "rx_rho", "tx_rho"),
    [("none", None, None), ("kronecker", 0.65, -0.35)],
)
def test_flat_mimo_empirical_covariance_matches_requested_model(
    execution_context, mode: str, rx_rho: float | None, tx_rho: float | None
) -> None:
    config = FlatMIMOFadingConfig(
        num_rx=1,
        num_rx_antennas=3,
        num_tx=1,
        num_tx_antennas=2,
        average_power_linear=1.0,
        power_normalization="per_coefficient",
        correlation_mode=mode,
        rx_correlation_coefficient=rx_rho,
        tx_correlation_coefficient=tx_rho,
        per_column_rx_correlation_coefficients=None,
    )
    request = ChannelSamplingRequest(
        30_000, 1, 1.0, execution_context, execution_context.random_stream(70_003)
    )
    matrices = as_numpy(FlatMIMOFading(config).realize(request).coefficients)[:, 0, :, 0, :]
    flattened = matrices.reshape(matrices.shape[0], -1)
    observed = flattened.conj().T @ flattened / flattened.shape[0]
    if mode == "none":
        expected = np.eye(6)
    else:
        receive = np.fromfunction(lambda i, j: rx_rho ** np.abs(i - j), (3, 3))
        transmit = np.fromfunction(lambda i, j: tx_rho ** np.abs(i - j), (2, 2))
        expected = np.kron(receive, transmit)
    assert np.max(np.abs(observed - expected)) < 0.035
    assert_unit_circularity(flattened)
