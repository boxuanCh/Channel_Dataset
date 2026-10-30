from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.fading import (
    FlatMIMOFading,
    FlatMIMOFadingConfig,
    parse_flat_mimo_config,
    register_flat_mimo_model,
)
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[3]


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def _config(normalization: str = "per_coefficient") -> FlatMIMOFadingConfig:
    return FlatMIMOFadingConfig(
        num_rx=2,
        num_rx_antennas=2,
        num_tx=3,
        num_tx_antennas=2,
        average_power_linear=2.0,
        power_normalization=normalization,
        correlation_mode="none",
        rx_correlation_coefficient=None,
        tx_correlation_coefficient=None,
        per_column_rx_correlation_coefficients=None,
    )


def _request(execution_context, batch_size: int, seed: int = 5) -> ChannelSamplingRequest:
    return ChannelSamplingRequest(
        batch_size=batch_size,
        num_time_samples=1,
        sampling_frequency_hz=1.0,
        execution=execution_context,
        random_stream=execution_context.random_stream(seed),
        request_id="flat-test",
    )


def test_flat_fading_shape_metadata_and_seeded_replay(execution_context) -> None:
    model = FlatMIMOFading(_config())
    first = model.realize(_request(execution_context, 4, seed=11))
    second = model.realize(_request(execution_context, 4, seed=11))
    assert first.shape == (4, 2, 2, 3, 2)
    assert first.metadata.model_name == "flat_mimo_fading"
    assert first.metadata.request_id == "flat-test"
    assert np.array_equal(_as_numpy(first.coefficients), _as_numpy(second.coefficients))


def test_per_coefficient_power_normalization(execution_context) -> None:
    response = FlatMIMOFading(_config()).realize(_request(execution_context, 30_000))
    empirical = np.mean(np.abs(_as_numpy(response.coefficients)) ** 2)
    assert empirical == pytest.approx(2.0, rel=0.025)


def test_per_link_power_normalization(execution_context) -> None:
    config = replace(_config("per_link"), num_rx=1, num_tx=1)
    response = FlatMIMOFading(config).realize(_request(execution_context, 30_000))
    matrices = _as_numpy(response.coefficients)[:, 0, :, 0, :]
    empirical = np.mean(np.sum(np.abs(matrices) ** 2, axis=(-2, -1)))
    assert empirical == pytest.approx(2.0, rel=0.025)


def test_batch_entries_are_independent(execution_context) -> None:
    config = replace(_config(), num_rx=1, num_tx=1, num_rx_antennas=1, num_tx_antennas=1)
    values = _as_numpy(
        FlatMIMOFading(config).realize(_request(execution_context, 20_000)).coefficients
    ).reshape(-1)
    correlation = np.corrcoef(values[:-1].real, values[1:].real)[0, 1]
    assert abs(correlation) < 0.03


def test_profile_constructs_registered_flat_model() -> None:
    envelope = load_config(ROOT / "configs" / "wireless" / "flat_mimo_fading.yaml")
    registry = ModelRegistry()
    register_flat_mimo_model(registry)
    model = registry.create(envelope)
    assert isinstance(model, FlatMIMOFading)
    assert model.config.correlation_mode == "kronecker"
    assert model.config.power_normalization == "per_link"


def test_flat_config_rejects_invalid_correlation_combinations() -> None:
    with pytest.raises(ValueError, match="requires receive and transmit"):
        replace(_config(), correlation_mode="kronecker")
    with pytest.raises(ValueError, match="must match num_tx_antennas"):
        replace(
            _config(),
            correlation_mode="per_column_receive",
            per_column_rx_correlation_coefficients=(0.5,),
        )


def test_flat_config_parser_rejects_unknown_fields() -> None:
    parameters = {
        "num_rx": 1,
        "num_rx_antennas": 1,
        "num_tx": 1,
        "num_tx_antennas": 1,
        "average_power_linear": 1.0,
        "power_normalization": "per_coefficient",
        "correlation_mode": "none",
        "rx_correlation_coefficient": None,
        "tx_correlation_coefficient": None,
        "per_column_rx_correlation_coefficients": None,
        "power_db": 0.0,
    }
    with pytest.raises(ValueError, match="unknown flat-MIMO"):
        parse_flat_mimo_config(parameters)
