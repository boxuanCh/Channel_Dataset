from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from channel_dataset.channels.wireless import (
    FlatChannelResponse,
    OFDMChannelResponse,
    TimeChannelResponse,
)


def _shares_storage(original, selected) -> bool:
    if type(original).__module__.partition(".")[0] == "torch":
        return original.untyped_storage().data_ptr() == selected.untyped_storage().data_ptr()
    return bool(np.shares_memory(original, selected))


def test_response_shapes_and_link_views(backend, realization_metadata) -> None:
    time_values = backend.ones((2, 3, 2, 4, 2, 5, 6), dtype="complex128")
    flat_values = backend.ones((2, 3, 2, 4, 2), dtype="complex128")
    ofdm_values = backend.ones((2, 3, 2, 4, 2, 5, 6), dtype="complex128")
    time = TimeChannelResponse(time_values, realization_metadata)
    flat = FlatChannelResponse(flat_values, realization_metadata)
    ofdm = OFDMChannelResponse(ofdm_values, realization_metadata)

    selected_time = time.select_link(1, 2, 3)
    selected_flat = flat.select_link(1, 2, 3)
    selected_ofdm = ofdm.select_link(1, 2, 3)
    assert selected_time.shape == (1, 1, 2, 1, 2, 5, 6)
    assert selected_flat.shape == (1, 1, 2, 1, 2)
    assert selected_ofdm.shape == (1, 1, 2, 1, 2, 5, 6)
    assert _shares_storage(time_values, selected_time.coefficients)
    assert _shares_storage(flat_values, selected_flat.coefficients)
    assert _shares_storage(ofdm_values, selected_ofdm.coefficients)


def test_responses_reject_rank_and_dtype(backend, realization_metadata) -> None:
    with pytest.raises(ValueError, match="rank 7"):
        TimeChannelResponse(
            backend.ones((1, 1, 1), dtype="complex128"), realization_metadata
        )
    with pytest.raises(TypeError, match="complex"):
        FlatChannelResponse(
            backend.ones((1, 1, 1, 1, 1), dtype="float64"), realization_metadata
        )


def test_response_objects_are_frozen(backend, realization_metadata) -> None:
    response = FlatChannelResponse(
        backend.ones((1, 1, 1, 1, 1), dtype="complex128"), realization_metadata
    )
    with pytest.raises(FrozenInstanceError):
        response.coefficients = None


def test_time_response_rejects_noninteger_lag_origin(backend, realization_metadata) -> None:
    values = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    with pytest.raises(TypeError, match="lag_min"):
        TimeChannelResponse(values, realization_metadata, lag_min=0.5)
