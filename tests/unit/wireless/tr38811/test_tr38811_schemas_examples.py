from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811TDLChannel,
    TR38811TDLConfig,
    register_tr38811_models,
)
from channel_dataset.config import load_config, load_resource_document
from channel_dataset.core import ExecutionContext
from channel_dataset.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[4]

SCHEMAS = (
    "tr38811-flat-v1.schema.json",
    "tr38811-tdl-v1.schema.json",
    "tr38811-cdl-v1.schema.json",
    "tr38811-system-v1.schema.json",
)


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize("schema_name", SCHEMAS)
def test_public_and_packaged_schemas_match(schema_name: str) -> None:
    public = json.loads(
        (ROOT / "configs/schema" / schema_name).read_text(encoding="utf-8")
    )
    packaged = load_resource_document(f"schema/{schema_name}")
    assert public == packaged
    jsonschema.validators.validator_for(public).check_schema(public)


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "filename,schema_name",
    [
        ("ngso_s_tdl.yaml", "tr38811-tdl-v1.schema.json"),
        ("geo_ka_flat.yaml", "tr38811-flat-v1.schema.json"),
        ("haps_system.yaml", "tr38811-system-v1.schema.json"),
    ],
)
def test_examples_validate_and_construct(filename: str, schema_name: str) -> None:
    envelope = load_config(ROOT / "configs/wireless/tr38811" / filename)
    jsonschema.validate(
        dict(envelope.parameters), load_resource_document(f"schema/{schema_name}")
    )
    registry = ModelRegistry()
    register_tr38811_models(registry)
    assert registry.create(envelope).name == envelope.name


@pytest.mark.ntn
@pytest.mark.tr38811
def test_deployment_frequency_and_bandwidth_are_enforced() -> None:
    valid = TR38811TDLConfig(
        "NTN-TDL-A", "15.4.0", "D3", "rural", 2.0e9, 20.0e6,
        100.0e-9, 1, 1,
    )
    with pytest.raises(ValueError, match="applicability range"):
        TR38811TDLChannel(replace(valid, carrier_frequency_hz=20.0e9))
    with pytest.raises(ValueError, match="maximum bandwidth"):
        TR38811TDLChannel(replace(valid, bandwidth_hz=21.0e6))


@pytest.mark.ntn
@pytest.mark.tr38811
def test_config_boolean_fields_are_strict() -> None:
    with pytest.raises(TypeError, match="Boolean"):
        TR38811TDLConfig(
            "NTN-TDL-A", "15.4.0", "D3", "rural", 2.0e9, 20.0e6,
            100.0e-9, 1, 1, apply_large_scale=1,  # type: ignore[arg-type]
        )


@pytest.mark.ntn
@pytest.mark.tr38811
def test_runtime_topology_must_match_deployment(
    zenith_topology_factory,
) -> None:
    execution = ExecutionContext("numpy", "double", "cpu")
    backend = get_backend("numpy")
    sampling = ChannelSamplingRequest(
        1, 1, 1000.0, execution, backend.random_stream(9)
    )
    config = TR38811TDLConfig(
        "NTN-TDL-A", "15.4.0", "D3", "rural", 2.0e9, 20.0e6,
        100.0e-9, 1, 1,
    )
    channel = TR38811TDLChannel(config)
    with pytest.raises(ValueError, match="satellite platform"):
        channel.realize(
            NTNChannelSamplingRequest(
                sampling,
                replace(zenith_topology_factory(execution), tx_endpoint_kind="haps"),
            )
        )
    with pytest.raises(ValueError, match="altitude"):
        channel.realize(
            NTNChannelSamplingRequest(
                sampling,
                zenith_topology_factory(execution, altitude_m=100_000.0),
            )
        )
