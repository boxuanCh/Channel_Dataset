import numpy as np
import pytest


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_seeded_stream_replays(random_stream) -> None:
    first = random_stream.fork("component").normal((32,), dtype="float64")
    second = random_stream.fork("component").normal((32,), dtype="float64")
    assert np.array_equal(_as_numpy(first), _as_numpy(second))


def test_child_streams_are_label_stable_and_independent(random_stream) -> None:
    first_a = _as_numpy(random_stream.fork("a").uniform((32,), dtype="float64"))
    first_b = _as_numpy(random_stream.fork("b").uniform((32,), dtype="float64"))
    second_b = _as_numpy(random_stream.fork("b").uniform((32,), dtype="float64"))
    assert not np.array_equal(first_a, first_b)
    assert np.array_equal(first_b, second_b)


def test_distribution_methods(random_stream) -> None:
    assert _as_numpy(random_stream.integers((8,), 0, 4)).shape == (8,)
    assert _as_numpy(random_stream.bernoulli((8,), 0.5)).dtype == np.bool_
    assert sorted(_as_numpy(random_stream.permutation(8)).tolist()) == list(range(8))


@pytest.mark.numpy
def test_numpy_stream_does_not_advance_global_rng() -> None:
    from channel_dataset.backends.numpy_backend import NumpyRandomStream

    np.random.seed(91)
    expected = np.random.random(4)
    np.random.seed(91)
    NumpyRandomStream(3).normal((100,))
    actual = np.random.random(4)
    assert np.array_equal(actual, expected)


@pytest.mark.torch
def test_torch_stream_does_not_advance_global_rng() -> None:
    torch = pytest.importorskip("torch")
    from channel_dataset.backends.torch_backend import TorchRandomStream

    torch.manual_seed(91)
    expected = torch.rand(4)
    torch.manual_seed(91)
    TorchRandomStream(3).normal((100,))
    actual = torch.rand(4)
    assert torch.equal(actual, expected)
