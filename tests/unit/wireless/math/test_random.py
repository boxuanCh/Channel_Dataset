import numpy as np
import pytest

from channel_dataset.channels.wireless.math import complex_normal


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_complex_normal_variance_convention(random_stream, execution_context) -> None:
    samples = _as_numpy(
        complex_normal(
            random_stream, (100_000,), variance=2.0, dtype=execution_context.complex_dtype
        )
    )
    assert abs(samples.mean()) < 0.02
    assert np.mean(np.abs(samples) ** 2) == pytest.approx(2.0, rel=0.02)
