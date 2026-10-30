from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.transforms import normalize_response
from channel_dataset.channels.wireless.types import (
    FlatChannelResponse,
    OFDMChannelResponse,
    TimeChannelResponse,
)
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


@pytest.mark.parametrize(
    ("response_type", "shape"),
    [
        (FlatChannelResponse, (2, 2, 2, 2, 2)),
        (TimeChannelResponse, (2, 2, 2, 2, 2, 3, 4)),
        (OFDMChannelResponse, (2, 2, 2, 2, 2, 3, 4)),
    ],
)
def test_normalization_sets_each_link_power(backend, backend_name, response_type, shape) -> None:
    values = backend.full(shape, 2.0 + 0j, dtype="complex128")
    response = response_type(values, _metadata(backend_name))
    normalized = normalize_response(response, target_power=3.0)
    result = _as_numpy(normalized.coefficients)
    axes = (2, 4) if len(shape) == 5 else (2, 4, 5, 6)
    assert type(normalized) is response_type
    assert np.allclose(np.mean(np.abs(result) ** 2, axis=axes), 3.0)


def test_time_normalization_preserves_lag_origin(backend, backend_name) -> None:
    values = backend.ones((1, 1, 1, 1, 1, 2, 3), dtype="complex128")
    response = TimeChannelResponse(values, _metadata(backend_name), lag_min=-2)
    assert normalize_response(response).lag_min == -2


def test_normalization_rejects_zero_power(backend, backend_name) -> None:
    values = backend.zeros((1, 1, 1, 1, 1), dtype="complex128")
    with pytest.raises(ValueError, match="zero-power"):
        normalize_response(FlatChannelResponse(values, _metadata(backend_name)))


@pytest.mark.parametrize("target", [0.0, -1.0, float("inf"), float("nan")])
def test_normalization_rejects_invalid_target(backend, backend_name, target) -> None:
    values = backend.ones((1, 1, 1, 1, 1), dtype="complex128")
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_response(
            FlatChannelResponse(values, _metadata(backend_name)), target_power=target
        )


@pytest.mark.torch
def test_normalization_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor(
        [[[[[1.0 + 1.0j, 2.0 + 0j]]]]], dtype=torch.complex128, requires_grad=True
    )
    response = FlatChannelResponse(values, _metadata("torch"))
    normalize_response(response).coefficients.real.sum().backward()
    assert values.grad is not None
