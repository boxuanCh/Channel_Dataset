import numpy as np


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_deterministic_math_matches_reference(backend) -> None:
    left = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64")
    right = backend.array([[2.0], [1.0]], dtype="float64")
    assert np.allclose(_as_numpy(backend.matmul(left, right)), [[4.0], [10.0]])
    assert np.allclose(_as_numpy(backend.exp(backend.array([0.0], dtype="float64"))), [1.0])
