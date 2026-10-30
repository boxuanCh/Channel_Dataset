from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.transforms import cir_to_time
from channel_dataset.channels.wireless.types import CIRBatch
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


def _cir(backend, backend_name, coefficients, delays, mask=None) -> CIRBatch:
    values = backend.array(coefficients, dtype="complex128")
    values = backend.reshape(values, (1, 1, 1, 1, 1, len(delays), -1))
    delay_values = backend.array([[[delays]]], dtype="float64")
    path_mask = None if mask is None else backend.array([[[mask]]], dtype="bool")
    return CIRBatch(values, delay_values, _metadata(backend_name), path_mask)


def test_zero_delay_unit_path_is_discrete_impulse(backend, backend_name) -> None:
    cir = _cir(backend, backend_name, [[1.0, 1.0]], [0.0])
    response = cir_to_time(cir, 1.0, lag_min=-1, lag_max=1)
    assert response.shape == (1, 1, 1, 1, 1, 2, 3)
    assert response.lag_min == -1
    assert response.lag_max == 1
    expected = np.array([0.0, 1.0, 0.0]).reshape(1, 1, 1, 1, 1, 1, 3)
    assert np.allclose(_as_numpy(response.coefficients), expected, atol=1e-15)


def test_one_sample_delay_moves_impulse_to_lag_one(backend, backend_name) -> None:
    response = cir_to_time(
        _cir(backend, backend_name, [[2.0]], [0.25]),
        4.0,
        lag_min=0,
        lag_max=2,
    )
    assert np.allclose(
        _as_numpy(response.coefficients).reshape(-1), [0.0, 2.0, 0.0], atol=1e-15
    )


def test_fractional_delay_uses_normalized_sinc(backend, backend_name) -> None:
    response = cir_to_time(
        _cir(backend, backend_name, [[1.0]], [0.5]),
        1.0,
        lag_min=0,
        lag_max=1,
    )
    assert np.allclose(
        _as_numpy(response.coefficients).reshape(-1),
        [np.sinc(-0.5), np.sinc(0.5)],
        atol=1e-12,
    )


def test_mask_and_time_varying_phase_are_preserved(backend, backend_name) -> None:
    coefficients = [[1.0, 1j, -1.0, -1j], [9.0, 9.0, 9.0, 9.0]]
    cir = _cir(backend, backend_name, coefficients, [0.0, 0.0], [True, False])
    response = cir_to_time(cir, 8.0, lag_min=0, lag_max=0)
    assert np.allclose(
        _as_numpy(response.coefficients).reshape(-1), [1.0, 1j, -1.0, -1j]
    )


def test_automatic_bounds_cover_maximum_delay(backend, backend_name) -> None:
    response = cir_to_time(_cir(backend, backend_name, [[1.0]], [2.1]), 1.0)
    assert response.lag_min == -6
    assert response.lag_max == 9


def test_cir_to_time_rejects_partial_bounds(backend, backend_name) -> None:
    with pytest.raises(ValueError, match="provided together"):
        cir_to_time(_cir(backend, backend_name, [[1.0]], [0.0]), 1.0, lag_min=0)


@pytest.mark.torch
def test_cir_to_time_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    coefficients = torch.ones(
        (1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128, requires_grad=True
    )
    delays = torch.tensor([[[[0.25]]]], dtype=torch.float64, requires_grad=True)
    cir = CIRBatch(coefficients, delays, _metadata("torch"))
    response = cir_to_time(cir, 1.0, lag_min=0, lag_max=1)
    response.coefficients.real.sum().backward()
    assert coefficients.grad is not None
    assert delays.grad is not None


@pytest.mark.torch
def test_cir_to_time_numpy_torch_parity() -> None:
    torch = pytest.importorskip("torch")
    numpy_cir = CIRBatch(
        np.array([[[[[[[1.0 + 2.0j], [0.5 - 0.25j]]]]]]]),
        np.array([[[[0.25, 0.75]]]]),
        _metadata("numpy"),
    )
    torch_cir = CIRBatch(
        torch.tensor(numpy_cir.coefficients, dtype=torch.complex128),
        torch.tensor(numpy_cir.delays_s, dtype=torch.float64),
        _metadata("torch"),
    )
    numpy_response = cir_to_time(numpy_cir, 2.0, lag_min=-1, lag_max=3)
    torch_response = cir_to_time(torch_cir, 2.0, lag_min=-1, lag_max=3)
    assert np.allclose(
        numpy_response.coefficients,
        torch_response.coefficients.detach().cpu().numpy(),
        atol=1e-12,
    )
