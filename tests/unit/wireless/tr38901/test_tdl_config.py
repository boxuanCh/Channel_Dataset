from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.tr38901 import (
    TappedDelayLine,
    TDLConfig,
    parse_tdl_config,
    register_tdl_model,
)
from channel_dataset.config import parse_config_document
from channel_dataset.registry import ModelRegistry


def _parameters(**overrides):
    values = {
        "profile": "TDL-A",
        "standard_version": "16.1.0",
        "delay_spread_s": 100e-9,
        "carrier_frequency_hz": 3.5e9,
        "bandwidth_hz": 100e6,
        "speed_mps": 3.0,
        "num_rx": 1,
        "num_rx_antennas": 1,
        "num_tx": 1,
        "num_tx_antennas": 1,
        "num_sinusoids": 16,
        "k_factor_db": None,
        "correlation_mode": "none",
        "rx_correlation_coefficient": None,
        "tx_correlation_coefficient": None,
        "joint_correlation_matrix": None,
    }
    values.update(overrides)
    return values


def test_parse_tdl_config_returns_typed_value() -> None:
    config = parse_tdl_config(_parameters())
    assert isinstance(config, TDLConfig)
    assert config.profile == "TDL-A"


def test_registry_constructs_tdl_from_profile() -> None:
    envelope = parse_config_document(
        {
            "schema_version": 1,
            "model": {
                "family": "wireless",
                "name": "tdl",
                "profile": "package://profiles/wireless/tdl_c.yaml",
            },
        }
    )
    registry = ModelRegistry()
    register_tdl_model(registry)
    model = registry.create(envelope)
    assert isinstance(model, TappedDelayLine)
    assert model.config.profile == "TDL-C"


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": "invalid"},
        {"delay_spread_s": None},
        {"carrier_frequency_hz": 0.0},
        {"bandwidth_hz": float("inf")},
        {"speed_mps": -1.0},
        {"num_sinusoids": 0},
        {"k_factor_db": 3.0},
    ],
)
def test_tdl_config_rejects_invalid_parameters(overrides) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_tdl_config(_parameters(**overrides))


def test_fixed_profile_rejects_delay_scaling() -> None:
    with pytest.raises(ValueError, match="must be null"):
        parse_tdl_config(_parameters(profile="TDLA30"))


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown TDL"):
        parse_tdl_config({**_parameters(), "typo": 1})


def test_joint_correlation_is_validated() -> None:
    config = parse_tdl_config(
        _parameters(
            num_rx_antennas=2,
            correlation_mode="joint",
            joint_correlation_matrix=[[1.0, 0.5], [0.5, 1.0]],
        )
    )
    assert config.joint_correlation_matrix == ((1.0, 0.5), (0.5, 1.0))
