import numpy as np
import pytest

from channel_dataset.backends.numpy_backend import NumpyBackend


@pytest.mark.numpy
def test_numpy_backend_core_operations() -> None:
    backend = NumpyBackend()
    value = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    assert backend.shape(value) == (2, 2)
    assert np.allclose(backend.matmul(value, value), [[7.0, 10.0], [15.0, 22.0]])
    assert np.allclose(backend.mean(value, axis=0), [2.0, 3.0])
    assert np.allclose(backend.ifft(backend.fft(value)), value)
    eigenvalues, _ = backend.eigh(value + value.T)
    assert eigenvalues.shape == (2,)
    assert backend.cholesky(np.eye(2)).shape == (2, 2)
    assert np.array_equal(backend.where(value > 2.0, value, np.zeros_like(value)), [[0, 0], [3, 4]])
    assert np.array_equal(backend.clip(value, 2.0, 3.0), [[2, 2], [3, 3]])
    assert np.array_equal(backend.cumsum(value, axis=1), [[1, 3], [3, 7]])


@pytest.mark.numpy
def test_numpy_rejects_non_cpu_device() -> None:
    backend = NumpyBackend()
    with pytest.raises(ValueError, match="only"):
        backend.zeros((1,), dtype="float32", device="cuda")
