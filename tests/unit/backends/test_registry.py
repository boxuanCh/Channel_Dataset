import pytest

from channel_dataset.backends import backend_name_for_array, get_backend


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("other")


def test_backend_name_for_numpy_array() -> None:
    import numpy as np

    assert backend_name_for_array(np.zeros(1)) == "numpy"


def test_backend_name_rejects_unknown_array() -> None:
    with pytest.raises(TypeError, match="unsupported array"):
        backend_name_for_array([1.0])
