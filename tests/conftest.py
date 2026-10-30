from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38901 import CDLConfig
from channel_dataset.core import ExecutionContext, RealizationMetadata

_PYTEST_TEMP_PREFIX = "channel-dataset-pytest-"
_TEMP_ROOT_ENV = "CHANNEL_DATASET_PYTEST_TMPDIR"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Give each test process a private temporary directory.

    Pytest's Windows default is a shared ``pytest-of-<username>`` directory.
    A test run inside a sandbox can create that directory with an ACL that the
    interactive Windows user cannot access.  A unique base directory avoids
    sharing ACL-protected state between those security identities.

    An explicit ``--basetemp`` remains authoritative.  The parent directory
    for automatically allocated paths can be overridden with
    ``CHANNEL_DATASET_PYTEST_TMPDIR``.
    """

    if config.option.basetemp is not None:
        return

    configured_parent = os.environ.get(_TEMP_ROOT_ENV)
    parent = (
        Path(configured_parent).expanduser()
        if configured_parent
        else Path(tempfile.gettempdir())
    )
    basetemp = parent / f"{_PYTEST_TEMP_PREFIX}{uuid.uuid4().hex}"
    config.option.basetemp = str(basetemp)
    config._channel_dataset_basetemp = basetemp  # type: ignore[attr-defined]


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove only the private temporary directory allocated above."""

    basetemp = getattr(config, "_channel_dataset_basetemp", None)
    if basetemp is not None:
        shutil.rmtree(basetemp, ignore_errors=True)


@pytest.fixture(params=["numpy", "torch"])
def backend_name(request: pytest.FixtureRequest) -> str:
    if request.param == "torch" and importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    return str(request.param)


@pytest.fixture
def backend(backend_name: str):
    return get_backend(backend_name)


@pytest.fixture
def execution_context(backend_name: str) -> ExecutionContext:
    return ExecutionContext(backend=backend_name, precision="double", device="cpu")


@pytest.fixture
def random_stream(execution_context: ExecutionContext):
    return execution_context.random_stream(12345)


@pytest.fixture
def realization_metadata(backend_name: str) -> RealizationMetadata:
    return RealizationMetadata(
        model_family="wireless",
        model_name="phase2_test",
        schema_version=1,
        backend=backend_name,
        precision="double",
        device="cpu",
        stream_id=f"{backend_name}:1:root",
        config_digest="0" * 64,
        resolved_configuration={},
        software_version="test",
    )


@pytest.fixture
def cdl_config_factory():
    """Build compact SISO CDL configurations for cross-package tests."""

    def factory(profile="CDL-A", **overrides):
        values = {
            "profile": profile,
            "standard_version": "16.1.0",
            "delay_spread_s": 1.0e-7,
            "carrier_frequency_hz": 3.5e9,
            "bandwidth_hz": 1.0e8,
            "link_direction": "downlink",
            "base_station_array_profile": "package://profiles/wireless/antenna_isotropic_siso.yaml",
            "terminal_array_profile": "package://profiles/wireless/antenna_isotropic_siso.yaml",
            "base_station_orientation_rad": (0.0, 0.0, 0.0),
            "terminal_orientation_rad": (0.0, 0.0, 0.0),
            "base_station_velocity_mps": (0.0, 0.0, 0.0),
            "terminal_velocity_mps": (0.0, 0.0, 0.0),
            "link_distance_m": 100.0,
            "k_factor_db": None,
        }
        values.update(overrides)
        return CDLConfig(**values)

    return factory
