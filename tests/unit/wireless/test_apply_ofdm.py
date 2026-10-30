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


def test_apply_ofdm_matches_hand_calculation(backend, backend_name) -> None:
    h = backend.array([[[[[[[2.0 + 0j]]], [[[3.0 + 0j]]]]]]], dtype="complex128")
    h = backend.reshape(h, (1, 1, 1, 1, 2, 1, 1))
    signal = backend.array([[[[[4.0 + 0j]], [[5.0 + 0j]]]]], dtype="complex128")
    signal = backend.reshape(signal, (1, 1, 2, 1, 1))
    output = apply_ofdm(OFDMChannelResponse(h, _metadata(backend_name)), signal)
    assert np.allclose(_as_numpy(output), 23.0)


@pytest.mark.torch
def test_apply_ofdm_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    h = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128)
    signal = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex128, requires_grad=True)
    response = OFDMChannelResponse(h, _metadata("torch"))
    output = apply_ofdm(response, signal)
    output.real.sum().backward()
    assert signal.grad is not None


def test_apply_ofdm_rejects_wrong_shape(backend, backend_name) -> None:
    h = backend.ones((1, 1, 1, 1, 1, 2, 3), dtype="complex128")
    signal = backend.ones((1, 1, 1, 1, 3), dtype="complex128")
    with pytest.raises(ValueError, match="does not match"):
        apply_ofdm(OFDMChannelResponse(h, _metadata(backend_name)), signal)
