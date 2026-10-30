from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38811 import (
    NTNChannelSamplingRequest,
    TR38811TDLChannel,
    TR38811TDLConfig,
    load_ntn_realization,
    ntn_to_ofdm,
    save_ntn_realization,
)
from channel_dataset.core import ExecutionContext


def _realization(execution: ExecutionContext, zenith_topology_factory):
    backend = get_backend(execution.backend)
    config = TR38811TDLConfig(
        "NTN-TDL-A", "15.4.0", "D3", "rural", 2e9, 20e6, 100e-9,
        1, 1, num_sinusoids=16, apply_large_scale=False,
    )
    sampling = ChannelSamplingRequest(1, 2, 1000.0, execution, backend.random_stream(19))
    return TR38811TDLChannel(config).realize(
        NTNChannelSamplingRequest(sampling, zenith_topology_factory(execution))
    )


@pytest.mark.ntn
@pytest.mark.tr38811
def test_ntn_persistence_round_trip(
    tmp_path: Path, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend="numpy", precision="double", device="cpu")
    original = _realization(execution, zenith_topology_factory)
    save_ntn_realization(tmp_path / "ntn", original)
    loaded = load_ntn_realization(tmp_path / "ntn", execution)
    assert np.array_equal(loaded.cir.coefficients, original.cir.coefficients)
    assert np.array_equal(loaded.geometry.slant_range_m, original.geometry.slant_range_m)
    assert np.array_equal(loaded.large_scale.total_db, original.large_scale.total_db)


@pytest.mark.ntn
@pytest.mark.tr38811
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_absolute_ofdm_adds_common_delay_phase(
    backend_name: str, zenith_topology_factory,
) -> None:
    execution = ExecutionContext(backend=backend_name, precision="double", device="cpu")
    backend = get_backend(backend_name)
    realization = _realization(execution, zenith_topology_factory)
    frequencies = backend.array([1000.0], dtype=execution.float_dtype, device=execution.device)
    synchronized = ntn_to_ofdm(realization, frequencies)
    absolute = ntn_to_ofdm(
        realization, frequencies, include_propagation_delay=True
    )
    delay = realization.geometry.propagation_delay_s[0, 0, 0, 0]
    expected = backend.exp(-2j * np.pi * frequencies[0] * delay)
    ratio = absolute.coefficients / synchronized.coefficients
    value = ratio if backend_name == "numpy" else ratio.detach().numpy()
    expected_value = expected if backend_name == "numpy" else expected.detach().numpy()
    assert np.allclose(value, expected_value)
