import numpy as np
import pytest

from channel_dataset.channels.wireless.transforms import subcarrier_frequencies


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_even_and_odd_centered_subcarriers(execution_context) -> None:
    even = _as_numpy(subcarrier_frequencies(4, 15.0, execution_context))
    odd = _as_numpy(subcarrier_frequencies(5, 15.0, execution_context))
    assert np.array_equal(even, [-30.0, -15.0, 0.0, 15.0])
    assert np.array_equal(odd, [-30.0, -15.0, 0.0, 15.0, 30.0])


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("inf"), float("nan")])
def test_subcarrier_frequencies_reject_invalid_spacing(execution_context, spacing) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        subcarrier_frequencies(4, spacing, execution_context)


def test_subcarrier_frequencies_reject_boolean_spacing(execution_context) -> None:
    with pytest.raises(TypeError, match="real number"):
        subcarrier_frequencies(4, True, execution_context)
