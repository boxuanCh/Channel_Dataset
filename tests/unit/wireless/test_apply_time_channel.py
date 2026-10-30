from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.operators import apply_time
from channel_dataset.channels.wireless.types import TimeChannelResponse
from channel_dataset.core import RealizationMetadata


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def _metadata(backend_name: str) -> RealizationMetadata:
    return RealizationMetadata(
        model_family="wireless",
        model_name="analytic",
        schema_version=1,
        backend=backend_name,
        precision="double",
        device="cpu",
        stream_id="test",
        config_digest="0" * 64,
        resolved_configuration={},
        software_version="test",
    )


def test_apply_time_uses_zero_padded_convolution_boundaries(backend, backend_name) -> None:
    h = backend.zeros((1, 1, 1, 1, 1, 4, 2), dtype="complex128")
    h[..., 1] = 1.0
    response = TimeChannelResponse(h, _metadata(backend_name), lag_min=0)
    signal = backend.array([1.0, 2.0, 3.0, 4.0], dtype="complex128")
    signal = backend.reshape(signal, (1, 1, 1, 4))
    assert np.allclose(_as_numpy(apply_time(response, signal)).reshape(-1), [0, 1, 2, 3])


def test_apply_time_supports_negative_lags(backend, backend_name) -> None:
    h = backend.ones((1, 1, 1, 1, 1, 3, 1), dtype="complex128")
    signal = backend.reshape(
        backend.array([1.0, 2.0, 3.0], dtype="complex128"), (1, 1, 1, 3)
    )
    response = TimeChannelResponse(h, _metadata(backend_name), lag_min=-1)
    assert np.allclose(_as_numpy(apply_time(response, signal)).reshape(-1), [2, 3, 0])


def test_apply_time_sums_transmit_antennas(backend, backend_name) -> None:
    h = backend.ones((1, 1, 1, 1, 2, 2, 1), dtype="complex128")
    signal = backend.reshape(
        backend.array([1.0, 2.0, 3.0, 4.0], dtype="complex128"), (1, 1, 2, 2)
    )
    output = apply_time(TimeChannelResponse(h, _metadata(backend_name)), signal)
    assert np.allclose(_as_numpy(output).reshape(-1), [4, 6])


def test_apply_time_rejects_wrong_shape(backend, backend_name) -> None:
    h = backend.ones((1, 1, 1, 1, 1, 3, 1), dtype="complex128")
    signal = backend.ones((1, 1, 1, 2), dtype="complex128")
    with pytest.raises(ValueError, match="does not match"):
        apply_time(TimeChannelResponse(h, _metadata(backend_name)), signal)


@pytest.mark.torch
def test_apply_time_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    h = torch.ones((1, 1, 1, 1, 1, 3, 1), dtype=torch.complex128, requires_grad=True)
    signal = torch.ones((1, 1, 1, 3), dtype=torch.complex128, requires_grad=True)
    output = apply_time(TimeChannelResponse(h, _metadata("torch")), signal)
    output.real.sum().backward()
    assert h.grad is not None
    assert signal.grad is not None
