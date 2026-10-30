import importlib.util

import pytest


@pytest.mark.cuda
@pytest.mark.torch
def test_cuda_backend_smoke_and_gradient():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("a CUDA-capable PyTorch runner is not available")
    from channel_dataset.backends import get_backend

    backend = get_backend("torch")
    value = backend.ones((4,), dtype="float32", device="cuda:0")
    value.requires_grad_(True)
    loss = backend.sum(value * value)
    loss.backward()
    assert value.device.type == "cuda"
    assert value.grad is not None
