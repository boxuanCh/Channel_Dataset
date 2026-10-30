from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.operators import apply_ofdm
from channel_dataset.channels.wireless.types import OFDMChannelResponse
from channel_dataset.core import RealizationMetadata


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _metadata(backend_name: str) -> RealizationMetadata:
    return RealizationMetadata(
        model_family="wireless", model_name="analytic", schema_version=1,
        backend=backend_name, precision="double", device="cpu", stream_id="test",
        config_digest="0" * 64, resolved_configuration={}, software_version="test",
    )


def test_constant_flat_mimo_path_across_grid(backend, backend_name) -> None:
    h = backend.full((1, 1, 1, 1, 1, 2, 3), 2.0, dtype="complex128")
    signal = backend.ones((1, 1, 1, 2, 3), dtype="complex128")
    output = apply_ofdm(OFDMChannelResponse(h, _metadata(backend_name)), signal)
    assert np.allclose(_as_numpy(output), 2.0)


def test_apply_ofdm_rejects_real_signal(backend, backend_name) -> None:
    h = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    signal = backend.ones((1, 1, 1, 1, 1), dtype="float64")
    with pytest.raises(TypeError, match="complex"):
        apply_ofdm(OFDMChannelResponse(h, _metadata(backend_name)), signal)
