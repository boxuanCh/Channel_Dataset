from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from channel_dataset.channels.wireless.math import matrix_sqrt_psd


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def test_diagonal_psd_square_root(backend) -> None:
    matrix = backend.array([[4.0, 0.0], [0.0, 9.0]], dtype="float64")
    root = matrix_sqrt_psd(matrix)
    assert np.allclose(_as_numpy(root), [[2.0, 0.0], [0.0, 3.0]])
    assert np.allclose(_as_numpy(backend.matmul(root, root)), _as_numpy(matrix))


def test_complex_psd_square_root_reconstructs_matrix(backend) -> None:
    matrix = backend.array([[1.0, 0.5j], [-0.5j, 1.0]], dtype="complex128")
    root = matrix_sqrt_psd(matrix)
    reconstructed = backend.matmul(root, root)
    assert np.allclose(_as_numpy(root), _as_numpy(root).conj().T, atol=1.0e-12)
    assert np.allclose(_as_numpy(reconstructed), _as_numpy(matrix), atol=1.0e-12)


def test_batched_semidefinite_square_root(backend) -> None:
    matrix = backend.array(
        [[[1.0, 1.0], [1.0, 1.0]], [[4.0, 0.0], [0.0, 0.0]]], dtype="float64"
    )
    root = matrix_sqrt_psd(matrix)
    assert np.allclose(_as_numpy(backend.matmul(root, root)), _as_numpy(matrix), atol=1.0e-12)


def test_nonhermitian_matrix_is_rejected(backend) -> None:
    matrix = backend.array([[1.0, 1.0], [0.0, 1.0]], dtype="float64")
    with pytest.raises(ValueError, match="Hermitian"):
        matrix_sqrt_psd(matrix)


def test_indefinite_matrix_is_rejected(backend) -> None:
    matrix = backend.array([[1.0, 0.0], [0.0, -0.1]], dtype="float64")
    with pytest.raises(ValueError, match="positive semidefinite"):
        matrix_sqrt_psd(matrix)


@pytest.mark.torch
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_torch_psd_square_root_preserves_gradients() -> None:
    import torch

    diagonal = torch.tensor([2.0, 5.0], dtype=torch.float64, requires_grad=True)
    matrix = torch.diag(diagonal)
    root = matrix_sqrt_psd(matrix)
    root.sum().backward()
    assert diagonal.grad is not None
    assert torch.isfinite(diagonal.grad).all()
