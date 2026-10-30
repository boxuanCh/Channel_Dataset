from __future__ import annotations

import numpy as np
import pytest

from channel_dataset.channels.wireless.transforms import time_lag_bounds, time_lags
from channel_dataset.core import ExecutionContext


def _as_numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return value


def test_time_lag_bounds_include_sinc_margin() -> None:
    assert time_lag_bounds(2.1e-6, 1e6) == (-6, 9)
    assert time_lag_bounds(0.0, 1e6, sinc_half_width=0) == (0, 0)


def test_time_lags_are_inclusive(backend_name) -> None:
    execution = ExecutionContext(backend_name, "double")
    values = time_lags(-2, 2, execution)
    assert np.array_equal(_as_numpy(values), [-2, -1, 0, 1, 2])


@pytest.mark.parametrize(
    ("arguments", "exception"),
    [
        ((-1.0, 1.0), ValueError),
        ((0.0, 0.0), ValueError),
        ((float("inf"), 1.0), ValueError),
        ((True, 1.0), TypeError),
    ],
)
def test_time_lag_bounds_reject_invalid_values(arguments, exception) -> None:
    with pytest.raises(exception):
        time_lag_bounds(*arguments)


def test_time_lags_reject_empty_interval(execution_context) -> None:
    with pytest.raises(ValueError, match="lag_max"):
        time_lags(2, 1, execution_context)
