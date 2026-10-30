from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from channel_dataset.channels.wireless import FlatChannelResponse
from channel_dataset.channels.wireless.operators import apply_flat
from channel_dataset.core import RealizationMetadata


def _metadata(backend_name: str, precision: str = "double") -> RealizationMetadata:
    return RealizationMetadata(
        model_family="wireless",
        model_name="flat_test",
        schema_version=1,
        backend=backend_name,
        precision=precision,
        device="cpu",
        stream_id=f"{backend_name}:1:root",
        config_digest="0" * 64,
        resolved_configuration={},
        software_version="test",
    )


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_apply_flat_matches_hand_matrix_multiplication(backend) -> None:
    coefficients = backend.array(
        [[[[[1.0, 2.0]], [[3.0, 4.0]]]]], dtype="complex128"
    )
    response = FlatChannelResponse(coefficients, _metadata(backend.name))
    signal = backend.array(
        [[[[5.0, 1.0], [6.0, 2.0]]]], dtype="complex128"
    )
    output = apply_flat(response, signal)
    assert np.allclose(_as_numpy(output), [[[[17.0, 5.0], [39.0, 11.0]]]])


def test_apply_flat_sums_multiple_transmitters(backend) -> None:
    coefficients = backend.ones((1, 1, 1, 2, 1), dtype="complex128")
    response = FlatChannelResponse(coefficients, _metadata(backend.name))
    signal = backend.array([[[2.0], [3.0]]], dtype="complex128")
    assert np.allclose(_as_numpy(apply_flat(response, signal)), [[[5.0]]])


def test_apply_flat_validates_leading_shape(backend) -> None:
    response = FlatChannelResponse(
        backend.ones((1, 1, 1, 1, 2), dtype="complex128"), _metadata(backend.name)
    )
    signal = backend.ones((2, 1, 2), dtype="complex128")
    with pytest.raises(ValueError, match="leading shape"):
        apply_flat(response, signal)


@pytest.mark.torch
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_apply_flat_preserves_torch_gradients() -> None:
    import torch

    coefficients = torch.tensor(
        [[[[[1.0 + 0.0j, 2.0 + 0.0j]]]]], dtype=torch.complex128, requires_grad=True
    )
    signal = torch.tensor([[[[3.0 + 0.0j], [4.0 + 0.0j]]]], dtype=torch.complex128)
    response = FlatChannelResponse(coefficients, _metadata("torch"))
    loss = torch.abs(apply_flat(response, signal)).sum()
    loss.backward()
    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
