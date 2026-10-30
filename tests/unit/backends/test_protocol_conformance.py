import numpy as np


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().resolve_conj().cpu().numpy()
    return np.asarray(value)


def test_backend_protocol_surface(backend) -> None:
    zeros = backend.zeros((2, 2), dtype="float64")
    ones = backend.ones((2, 2), dtype="float64")
    full = backend.full((2, 2), 2.0, dtype="float64")
    sequence = backend.arange(4, dtype="float64")
    matrix = backend.reshape(sequence, (2, 2))

    assert backend.name in {"numpy", "torch"}
    assert backend.is_array(matrix)
    assert backend.ndim(matrix) == 2
    assert backend.shape(matrix) == (2, 2)
    assert backend.device(matrix) == "cpu"
    assert backend.is_floating(matrix)
    assert not backend.is_complex(matrix)
    assert backend.all_finite(matrix)
    assert backend.all_nonnegative(matrix)
    assert backend.dtype(backend.astype(matrix, "float32")) == backend.resolve_dtype("float32")
    assert np.array_equal(_as_numpy(backend.transpose(matrix, (1, 0))), [[0.0, 2.0], [1.0, 3.0]])
    assert _as_numpy(backend.broadcast_to(ones[:1, :1], (2, 2))).shape == (2, 2)
    assert backend.shape(backend.stack((zeros, ones), axis=0)) == (2, 2, 2)
    assert backend.shape(backend.concatenate((zeros, ones), axis=0)) == (4, 2)
    unsorted = backend.array([[2.0, 0.0, 1.0]], dtype="float64")
    order = backend.argsort(unsorted, axis=-1)
    assert np.array_equal(_as_numpy(backend.sort(unsorted)), [[0.0, 1.0, 2.0]])
    assert np.array_equal(
        _as_numpy(backend.take_along_axis(unsorted, order, axis=-1)),
        [[0.0, 1.0, 2.0]],
    )
    assert float(_as_numpy(backend.sum(ones))) == 4.0
    assert float(_as_numpy(backend.mean(full))) == 2.0
    assert float(_as_numpy(backend.min(matrix))) == 0.0
    assert float(_as_numpy(backend.max(matrix))) == 3.0

    positive = backend.array([1.0, 4.0], dtype="float64")
    assert np.allclose(_as_numpy(backend.abs(-positive)), [1.0, 4.0])
    assert np.allclose(_as_numpy(backend.sqrt(positive)), [1.0, 2.0])
    assert np.allclose(_as_numpy(backend.exp(zeros)), 1.0)
    assert np.allclose(_as_numpy(backend.log(positive)), np.log([1.0, 4.0]))
    assert np.allclose(_as_numpy(backend.log10(positive)), np.log10([1.0, 4.0]))
    assert np.allclose(_as_numpy(backend.sin(zeros)), 0.0)
    assert np.allclose(_as_numpy(backend.sinc(backend.array([0.0, 1.0], dtype="float64"))), [1.0, 0.0], atol=1e-15)
    assert np.allclose(_as_numpy(backend.cos(zeros)), 1.0)
    assert np.allclose(_as_numpy(backend.atan2(ones, ones)), np.pi / 4.0)

    complex_value = backend.array([1.0 + 2.0j], dtype="complex128")
    bool_value = backend.array([True], dtype="bool")
    integer_value = backend.array([3], dtype="int64")
    assert backend.is_complex(complex_value)
    assert backend.is_bool(bool_value)
    assert backend.is_integer(integer_value)
    assert backend.to_scalar(integer_value[0]) == 3
    assert np.array_equal(_as_numpy(backend.conj(complex_value)), [1.0 - 2.0j])
    assert np.allclose(_as_numpy(backend.matmul(ones, full)), 4.0)
    assert np.allclose(_as_numpy(backend.einsum("ij,jk->ik", ones, full)), 4.0)
    assert np.allclose(_as_numpy(backend.ifft(backend.fft(complex_value))), [1.0 + 2.0j])
    assert np.allclose(_as_numpy(backend.cholesky(backend.array(np.eye(2), dtype="float64"))), np.eye(2))
    eigenvalues, eigenvectors = backend.eigh(backend.array(np.eye(2), dtype="float64"))
    assert backend.shape(eigenvalues) == (2,)
    assert backend.shape(eigenvectors) == (2, 2)
    singular_values = backend.svdvals(backend.array(np.eye(2), dtype="float64"))
    assert np.allclose(_as_numpy(singular_values), [1.0, 1.0])
    assert backend.random_stream(5).backend_name == backend.name
