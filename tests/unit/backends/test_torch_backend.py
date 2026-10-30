import pytest

torch = pytest.importorskip("torch")

from channel_dataset.backends.torch_backend import TorchBackend


@pytest.mark.torch
def test_torch_backend_core_operations_and_gradient() -> None:
    backend = TorchBackend()
    value = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    value.requires_grad_(True)
    result = backend.sum(backend.matmul(value, value))
    result.backward()
    assert tuple(value.grad.shape) == (2, 2)
    assert torch.allclose(backend.ifft(backend.fft(value)), value.to(torch.complex128))
    assert backend.cholesky(torch.eye(2, dtype=torch.float64)).shape == (2, 2)
    eigenvalues, _ = backend.eigh(value.detach() + value.detach().T)
    assert eigenvalues.shape == (2,)
    zero = torch.zeros_like(value)
    assert torch.equal(backend.where(value > 2.0, value, zero), torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=value.dtype))
    assert torch.equal(backend.clip(value, 2.0, 3.0), torch.tensor([[2.0, 2.0], [3.0, 3.0]], dtype=value.dtype))
    assert torch.equal(backend.cumsum(value, axis=1), torch.tensor([[1.0, 3.0], [3.0, 7.0]], dtype=value.dtype))


@pytest.mark.torch
def test_unavailable_cuda_is_rejected() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(ValueError, match="unavailable"):
        TorchBackend().zeros((1,), dtype="float32", device="cuda")
