from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.transforms import cir_to_ofdm
from channel_dataset.channels.wireless.types import CIRBatch
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


def test_zero_delay_unit_path_is_flat(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 2), dtype="complex128")
    delays = backend.zeros((1, 1, 1, 1), dtype="float64")
    frequencies = backend.array([-1.0, 0.0, 1.0], dtype="float64")
    response = cir_to_ofdm(CIRBatch(coefficients, delays, _metadata(backend_name)), frequencies)
    assert response.shape == (1, 1, 1, 1, 1, 2, 3)
    assert np.allclose(_as_numpy(response.coefficients), 1.0)


def test_pure_delay_has_expected_phase(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    delays = backend.full((1, 1, 1, 1), 0.25, dtype="float64")
    frequencies = backend.array([0.0, 1.0], dtype="float64")
    response = cir_to_ofdm(CIRBatch(coefficients, delays, _metadata(backend_name)), frequencies)
    assert np.allclose(_as_numpy(response.coefficients).reshape(-1), [1.0, -1j], atol=1e-12)


def test_path_mask_excludes_invalid_path(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 2, 1), dtype="complex128")
    delays = backend.zeros((1, 1, 1, 2), dtype="float64")
    mask = backend.array([[[[True, False]]]], dtype="bool")
    frequencies = backend.array([0.0], dtype="float64")
    cir = CIRBatch(coefficients, delays, _metadata(backend_name), mask)
    assert np.allclose(_as_numpy(cir_to_ofdm(cir, frequencies).coefficients), 1.0)


def test_two_paths_construct_and_cancel(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 2, 1), dtype="complex128")
    delays = backend.array([[[[0.0, 0.5]]]], dtype="float64")
    frequencies = backend.array([0.0, 1.0], dtype="float64")
    response = cir_to_ofdm(CIRBatch(coefficients, delays, _metadata(backend_name)), frequencies)
    assert np.allclose(_as_numpy(response.coefficients).reshape(-1), [2.0, 0.0], atol=1e-12)


def test_time_varying_phase_ramp_becomes_symbol_axis(backend, backend_name) -> None:
    coefficients = backend.array([1.0, 1j, -1.0, -1j], dtype="complex128")
    coefficients = backend.reshape(coefficients, (1, 1, 1, 1, 1, 1, 4))
    delays = backend.zeros((1, 1, 1, 1), dtype="float64")
    frequencies = backend.array([0.0], dtype="float64")
    response = cir_to_ofdm(CIRBatch(coefficients, delays, _metadata(backend_name)), frequencies)
    assert np.allclose(_as_numpy(response.coefficients).reshape(-1), [1, 1j, -1, -1j])


def test_cir_to_ofdm_rejects_empty_frequencies(backend, backend_name) -> None:
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    delays = backend.zeros((1, 1, 1, 1), dtype="float64")
    frequencies = backend.array([], dtype="float64")
    with pytest.raises(ValueError, match="empty"):
        cir_to_ofdm(CIRBatch(coefficients, delays, _metadata(backend_name)), frequencies)


@pytest.mark.torch
def test_cir_to_ofdm_preserves_torch_gradients() -> None:
    torch = pytest.importorskip("torch")
    coefficients = torch.ones(
        (1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128, requires_grad=True
    )
    delays = torch.tensor([[[[0.25]]]], dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([1.0], dtype=torch.float64)
    response = cir_to_ofdm(CIRBatch(coefficients, delays, _metadata("torch")), frequencies)
    response.coefficients.real.sum().backward()
    assert coefficients.grad is not None
    assert delays.grad is not None
